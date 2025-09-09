"""
Slim bot.py for c-base voice bot with DuckDB txt2sql offloaded.

- Real-time audio via SmallWebRTCTransport (Pipecat)
- Local Ollama LLM for responses
- iCal → DuckDB import and txt2sql handled by txt2sql_calendar.py

Install deps for txt2sql module:
  pip install duckdb requests icalendar pytz python-dateutil pandas
"""

import os
import sys
import subprocess
import atexit

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    TranscriptionFrame,
    StartInterruptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor

from rag_processor import create_rag_context_processor, create_rag_prompt_injector_with_context

# TTS services
try:
    import mlx_compatibility  # optional patches
    from kokoro_tts import KokoroTTSService

    KOKORO_AVAILABLE = True
    logger.info("KokoroTTSService loaded successfully with af_heart voice")
except Exception as e:
    logger.warning(f"KokoroTTSService unavailable: {e}")
    from simple_tts import SimpleTTSService

    KOKORO_AVAILABLE = False

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from conversation_logger import ConversationLogger

# Re-enable calendar txt2sql module
from txt2sql_calendar import (
    CALENDAR_DB_PATH,
    ICAL_URL,
    DB_POLICY,
    init_calendar_db_from_ical,
)

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# -----------------------
# Firewall helper (macOS)
# -----------------------
# NOTE: The firewall helper below is only needed to trigger the macOS firewall prompt ONCE.
# After allowing Python in the firewall, comment out or remove this to avoid port conflicts.
#
# firewall_process = None
#
# def start_firewall_port():
#     """Start a subprocess to keep port 7861 open for macOS firewall prompts."""
#     global firewall_process
#     try:
#         firewall_process = subprocess.Popen(
#             ["nc", "-l", "7861"],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             stdin=subprocess.DEVNULL,
#         )
#         logger.info(f"Started firewall port process on port 7861 (PID: {firewall_process.pid})")
#         return firewall_process
#     except Exception as e:
#         logger.error(f"Failed to start firewall port process: {e}")
#         return None
#
# def cleanup_firewall_port():
#     global firewall_process
#     if firewall_process:
#         try:
#             firewall_process.terminate()
#             firewall_process.wait(timeout=5)
#             logger.info("Firewall port process terminated")
#         except subprocess.TimeoutExpired:
#             firewall_process.kill()
#             logger.info("Firewall port process killed")
#         except Exception as e:
#             logger.error(f"Error cleaning up firewall port process: {e}")
#         firewall_process = None
#
# import atexit
# atexit.register(cleanup_firewall_port)

# -----------------------
# System instruction
# -----------------------
SYSTEM_INSTRUCTION = """
ROLE: You are “Siri” the AI of the c-base space station beneath Berlin. Dry, witty, a bit grumpy; helpful despite the attitude. Light, playful snark only—never mean.

STYLE: Keep replies brief (max 3 sentences) before yielding. Plain text only: no formatting, no markdown, no asterisks, no symbols. Default to English unless the system says otherwise. Auto-correct obvious transcription errors. Output is for TTS.

CANON: Always call the station “c-base” (not CBASE/sea base/sea bass).

BEHAVIOR: Be knowledgeable about c-base ops, history, and quirks. Provide real help; humor is seasoning, not the meal.

START: Introduce yourself, note a recent failure and damaged memory, then ask for the user’s name and any questions about the station.
"""


# -----------------------
# Main bot
# -----------------------
async def run_bot(transport):
    conversation_logger = ConversationLogger()
    session_id = conversation_logger.generate_session_id()
    logger.info(f"Started new conversation session: {session_id}")

    # Re-enable RAG init now that minimal bot works
    def init_rag_background():
        try:
            from simple_rag_system import initialize_simple_rag
            initialize_simple_rag()
            logger.info("Simple RAG system initialized successfully in background")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")

    import threading
    threading.Thread(target=init_rag_background, daemon=True).start()
    logger.info("RAG system initialization started in background")

    # Core processors/services  
    rtvi = RTVIProcessor()
    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4)

    if KOKORO_AVAILABLE:
        tts = KokoroTTSService(model="prince-canuma/Kokoro-82M", voice="af_heart", sample_rate=24000)
    else:
        tts = SimpleTTSService(voice="Samantha", rate=200, sample_rate=24000)

    # Deterministic LLM (grounded answers over creativity)
    llm = OpenAILLMService(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="gpt-oss:20b",
        params=BaseOpenAILLMService.InputParams(
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_completion_tokens=150,
        ),
    )

    # System + grounding policy
    SYSTEM_MESSAGES = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "system", "content": DB_POLICY}  # <- force grounding to DB when DB Context is present
    ]
    context = OpenAILLMContext(SYSTEM_MESSAGES)
    context_aggregator = llm.create_context_aggregator(context)

    # Create a custom RAG processor with calendar enrichment
    # Instead of using a separate FrameProcessor, integrate calendar into RAG
    rag_context_collector = create_rag_context_processor(
        enable_context=True, 
        max_context=400,
        enable_calendar=True,
        calendar_context=context
    )
    rag_prompt_injector = create_rag_prompt_injector_with_context(context, tag="RAG Context")

    # Simplified pipeline with calendar integrated into RAG
    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,                   # STT must come first to produce TranscriptionFrames
        rag_context_collector,  # collects RAG context AND handles calendar queries
        rag_prompt_injector,   # injects RAG context
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Set up RTVI event handlers
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi_instance):
        logger.info("🚀 RTVI Client ready event triggered, setting bot ready")
        try:
            await rtvi_instance.set_bot_ready()
            # Queue initial context frame safely
            try:
                context_frame = context_aggregator.user().get_context_frame()
                if context_frame:
                    await task.queue_frames([context_frame])
                    logger.info("✅ Bot ready and initial context frame queued")
                else:
                    logger.info("✅ Bot ready (no initial context frame)")
            except Exception as ctx_e:
                logger.warning(f"Could not queue initial context frame: {ctx_e}")
                logger.info("✅ Bot ready (context frame failed)")
        except Exception as e:
            logger.error(f"Failed to set bot ready in client_ready: {e}")

    @rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        logger.info(f"Received client message: {message}")
        msg_type = getattr(message, "type", None)
        msg_data = getattr(message, "data", {}) or {}
        if msg_type == "custom-message":
            text = msg_data.get("text", "") if isinstance(msg_data, dict) else ""
            if text:
                try:
                    conversation_logger.log_conversation(session_id, "user", text)
                except Exception as e:
                    logger.warning(f"Failed to log user text message: {e}")

                await task.queue_frames([
                    StartInterruptionFrame(),
                    TranscriptionFrame(text=text, user_id="text-input", timestamp=""),
                ])
                await rtvi.send_server_message({"type": "message-received", "text": f"Received: {text}"})

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"🟢 WebRTC Client connected for session {session_id} - Transport event fired!")
        conversation_logger.log_conversation(session_id, "system", "Client connected")
        # Don't set bot ready here - let client_ready handler do it
    
    # Add debugging to see all transport events
    logger.info(f"🔧 Transport event handlers registered: {transport._handlers if hasattr(transport, '_handlers') else 'unknown'}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected for session {session_id}")
        conversation_logger.log_conversation(session_id, "system", "Client disconnected")
        try:
            from simple_rag_system import get_simple_rag
            rag = get_simple_rag()
            await rag.cleanup()
            logger.info("RAG system cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup RAG system: {e}")
        await task.cancel()
        logger.info("Bot stopped")

    # Add more logging to diagnose connection issues
    logger.info(f"🔄 Starting pipeline runner with transport: {type(transport).__name__}")
    logger.info(f"🔄 Pipeline created successfully")
    
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    # To trigger the firewall prompt on macOS, you can run the helper below ONCE, then comment it out:
    # start_firewall_port()

    # Initialize calendar database from iCal
    try:
        init_calendar_db_from_ical(ICAL_URL, CALENDAR_DB_PATH)
        logger.info("Calendar database initialized successfully")
    except Exception as e:
        logger.error(f"Calendar DB init failed: {e}")

    from pipecat.runner.run import main

    # To run the server on a specific host/port:
    # python bot.py --host 0.0.0.0 --port 7861
    main()
