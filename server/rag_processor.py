"""
RAG processors for Pipecat — no UI leaks, full RAG power.

Design:
- Never mutate user-visible text inside TranscriptionFrame.
- Collect RAG context and attach it to the frame (hidden).
- Inject the context as a SYSTEM message into OpenAILLMContext just before the user
  message is added — so the LLM sees it, but the UI never does.

Pipeline (important):
  ... -> STT ->
      RAGContextProcessor()             # attach hidden context to the frame
      RAGPromptInjectorWithContext(ctx) # push one-off SYSTEM message with the context
      context_aggregator.user()         # original user text only
      LLM -> ...
"""

import asyncio
from typing import Optional
from loguru import logger

from pipecat.frames.frames import (
    TranscriptionFrame,
    LLMTextFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

# Optional deps guarded to avoid hard crashes if modules are absent
try:
    from simple_rag_system import get_simple_rag, SimpleSpaceStationRAG  # type: ignore
except Exception:
    get_simple_rag = None
    class SimpleSpaceStationRAG:  # type: ignore
        pass

try:
    from lightrag_system import get_rag_system, SpaceStationRAG  # type: ignore
except Exception:
    get_rag_system = None
    class SpaceStationRAG:  # type: ignore
        pass

try:
    from translation_service import get_translation_service  # type: ignore
except Exception:
    def get_translation_service():
        return None

try:
    from multilingual_tts import get_multilingual_tts  # type: ignore
except Exception:
    def get_multilingual_tts():
        return None

# Calendar integration
try:
    from txt2sql_calendar import enrich_calendar_query  # type: ignore
except Exception:
    async def enrich_calendar_query(*args, **kwargs):
        return False


# =============================================================================
# Frame subtype: carries hidden RAG context; .text stays pristine for UI
# =============================================================================
class RAGAnnotatedTranscriptionFrame(TranscriptionFrame):
    def __init__(self, *, text: str, user_id: str = "", timestamp: str = "", rag_context: str = ""):
        super().__init__(text=text, user_id=user_id, timestamp=timestamp)
        self.rag_context = rag_context  # Hidden payload for prompt injector


# =============================================================================
# Helpers
# =============================================================================
_SHORT = {"hello", "hi", "hey", "yes", "no", "ok", "thanks", "thank you"}

def _should_skip(query: str) -> bool:
    q = (query or "").strip().lower()
    if len(q) < 10 or q in _SHORT:
        return True
    
    # Skip calendar questions - these should be handled by txt2sql, not RAG
    try:
        from txt2sql_calendar import looks_like_db_question
        if looks_like_db_question(query):
            return True
    except ImportError:
        pass
    
    return False


# =============================================================================
# Collector: fetch RAG context, attach to frame (DO NOT change .text)
# =============================================================================
class RAGContextProcessor(FrameProcessor):
    """
    Collects relevant RAG context for the user utterance and returns a
    RAGAnnotatedTranscriptionFrame that preserves the original text.

    Place BEFORE RAGPromptInjectorWithContext and BEFORE context_aggregator.user().
    """

    def __init__(
        self,
        rag_system: Optional["SimpleSpaceStationRAG"] = None,
        enable_context_injection: bool = True,
        enable_translation: bool = False,
        enable_calendar: bool = False,
        max_context_length: int = 800,
        context_threshold: float = 0.7,  # reserved for future filtering
        timeout_sec: float = 2.0,
        calendar_context = None,
    ):
        super().__init__()
        self.rag_system = rag_system or (get_simple_rag() if get_simple_rag else None)
        self.enable_context_injection = enable_context_injection
        self.enable_translation = enable_translation
        self.max_context_length = max_context_length
        self.context_threshold = context_threshold
        self.timeout_sec = timeout_sec
        self._started = False
        self.enable_calendar = enable_calendar
        self.calendar_context = calendar_context

        self.translation_service = get_translation_service() if enable_translation else None
        self.tts_service = get_multilingual_tts() if enable_translation else None

        logger.info(
            f"Initialized RAGContextProcessor(context_injection={enable_context_injection}, "
            f"translation={enable_translation}, calendar={enable_calendar}, max_context_length={max_context_length})"
        )

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if not self._started:
            if isinstance(frame, StartFrame):
                self._started = True
                logger.debug("RAGContextProcessor started")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame) and self.enable_context_injection:
            annotated = await self._attach_context(frame)
            await self.push_frame(annotated, direction)
        else:
            await self.push_frame(frame, direction)

    async def _attach_context(self, frame: TranscriptionFrame) -> TranscriptionFrame:
        user_query = (frame.text or "").strip()
        if _should_skip(user_query) or self.rag_system is None:
            return frame

        # Optional translation (non-fatal, best effort)
        if self.translation_service:
            try:
                user_query = self.translation_service.to_english(user_query) or user_query
            except Exception as e:
                logger.debug(f"Translation skipped: {e}")

        # Handle calendar queries if enabled
        if self.enable_calendar and self.calendar_context:
            try:
                calendar_handled = await enrich_calendar_query(
                    user_query, 
                    self.calendar_context, 
                    push_frame_func=self.push_frame
                )
                if calendar_handled:
                    # Calendar query was processed - context was enriched
                    logger.debug("Calendar query processed, skipping RAG")
                    return frame
            except Exception as e:
                logger.debug(f"Calendar processing skipped: {e}")

        # Get context with proper exception handling - avoid async conflicts
        try:
            # Force synchronous call to prevent event loop conflicts
            if hasattr(self.rag_system, 'search_similar'):
                # Use lower-level search method to avoid async conflicts
                search_results = self.rag_system.search_similar(user_query, top_k=2)
                if search_results:
                    context_parts = []
                    total_length = 0
                    for result in search_results:
                        content = result.get("content", "")
                        if total_length + len(content) <= self.max_context_length:
                            context_parts.append(content)
                            total_length += len(content)
                        else:
                            remaining = self.max_context_length - total_length
                            if remaining > 50:
                                context_parts.append(content[:remaining] + "...")
                            break
                    context = " ".join(context_parts)
                else:
                    context = ""
            else:
                # Fallback to original method if search_similar not available
                context = self.rag_system.get_relevant_context(user_query, self.max_context_length)
            logger.info(f"🔍 RAGContextProcessor: got {len(context or '')} chars of context for query: '{user_query[:50]}...'")
            if context:
                logger.info(f"🔍 Context preview: {context[:100]}...")
        except Exception as e:
            logger.debug(f"RAG context skipped (error): {e}")
            context = ""

        if context and context.strip():
            logger.info(f"🏷️ Creating RAGAnnotatedTranscriptionFrame with {len(context.strip())} chars context")
            return RAGAnnotatedTranscriptionFrame(
                text=frame.text,                 # keep original user-visible text
                user_id=frame.user_id,
                timestamp=frame.timestamp,
                rag_context=context.strip(),     # hidden payload for injector
            )
        logger.info(f"⚪ No context found, returning original TranscriptionFrame")
        return frame


# =============================================================================
# Injector: push a one-off SYSTEM message into OpenAILLMContext (LLM-only)
# =============================================================================
class RAGPromptInjectorWithContext(FrameProcessor):
    """
    If a frame carries rag_context, append a SYSTEM message to the provided
    OpenAILLMContext instance:

        [RAG Context: <context>]

    Then pass the ORIGINAL frame downstream unchanged so UI sees only user text.

    Place RIGHT BEFORE context_aggregator.user().
    """

    def __init__(self, openai_llm_context, tag: str = "RAG Context"):
        """
        :param openai_llm_context: the same OpenAILLMContext instance used by your bot
        :param tag: label used in the injected system message
        """
        super().__init__()
        self.ctx = openai_llm_context
        self.tag = tag
        self._started = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if not self._started:
            if isinstance(frame, StartFrame):
                self._started = True
                logger.debug("RAGPromptInjectorWithContext started")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, RAGAnnotatedTranscriptionFrame) and getattr(frame, "rag_context", ""):
            context = frame.rag_context
            try:
                # Append a SYSTEM message into the active OpenAILLMContext.
                # We do NOT change the user text. UI remains clean.
                system_message = {
                    "role": "system",
                    "content": f"[{self.tag}: {context}]"
                }
                self.ctx.messages.append(system_message)
                logger.info(f"✅ RAG CONTEXT INJECTED: {len(context)} chars into OpenAILLMContext")
                logger.info(f"✅ Context preview: {context[:100]}...")
                logger.info(f"✅ Total context messages: {len(self.ctx.messages)}")
            except Exception as e:
                logger.error(f"❌ Failed to inject system context: {e}")

            # Pass the original frame through unchanged
            await self.push_frame(TranscriptionFrame(text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp), direction)
            
        elif isinstance(frame, RAGAnnotatedTranscriptionFrame):
            logger.warning(f"⚠️ RAGAnnotatedTranscriptionFrame has no context: '{getattr(frame, 'rag_context', 'MISSING')}'")
            # Pass the original transcription frame through unchanged
            await self.push_frame(TranscriptionFrame(text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp), direction)
        else:
            # Pass all other frames through unchanged
            await self.push_frame(frame, direction)


# =============================================================================
# Optional: direct-answer processor (does not touch user text)
# =============================================================================
class RAGQueryProcessor(FrameProcessor):
    """
    For certain queries, respond straight from RAG with an LLMTextFrame,
    bypassing LLM generation. Does not mutate user-visible text.
    """

    def __init__(
        self,
        rag_system: Optional["SimpleSpaceStationRAG"] = None,
        direct_rag_keywords: Optional[list] = None,
    ):
        super().__init__()
        self.rag_system = rag_system or (get_simple_rag() if get_simple_rag else None)
        self.direct_rag_keywords = direct_rag_keywords or [
            "emergency procedure",
            "how to fix",
            "system status",
            "diagnostic",
            "manual",
            "specification",
            "technical details",
        ]
        self._started = False
        logger.info("Initialized RAGQueryProcessor for direct RAG queries")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if not self._started:
            if isinstance(frame, StartFrame):
                self._started = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            if await self._should_direct(frame.text):
                rag_response = await self._answer(frame.text)
                await self.push_frame(LLMTextFrame(text=rag_response), direction)
                return

        await self.push_frame(frame, direction)

    async def _should_direct(self, query: str) -> bool:
        q = (query or "").lower()
        return any(k in q for k in self.direct_rag_keywords)

    async def _answer(self, query: str) -> str:
        try:
            if self.rag_system is None:
                return "Sorry, my knowledge modules are offline."

            try:
                response = await self.rag_system.query_with_context_async(query)  # type: ignore[attr-defined]
            except AttributeError:
                response = self.rag_system.query_with_context(query)

            prefix = "Well, let me check my systems... "
            if "emergency" in query.lower():
                prefix = "Hold on, this is important. "
            elif "fix" in query.lower() or "repair" in query.lower():
                prefix = "Ah, a technical problem. Let me see... "
            return f"{prefix}{response}"
        except Exception as e:
            logger.error(f"Direct RAG query failed: {e}")
            return "Sorry, I'm having trouble accessing that information right now."


# =============================================================================
# LightRAG variant: same no-leak pattern (collector only)
# =============================================================================
class LightRAGContextProcessor(FrameProcessor):
    """
    Lightweight RAG collector using LightRAG.
    Attaches context via RAGAnnotatedTranscriptionFrame (no text mutation).
    """

    def __init__(
        self,
        rag_system: Optional["SpaceStationRAG"] = None,
        max_context_length: int = 400,
        enable_context_injection: bool = True,
        timeout_sec: float = 2.0,
    ):
        super().__init__()
        self.rag_system = rag_system or (get_rag_system() if get_rag_system else None)
        self.max_context_length = max_context_length
        self.enable_context_injection = enable_context_injection
        self.timeout_sec = timeout_sec
        self._started = False

        logger.info(f"Initialized LightRAGContextProcessor (max_context={max_context_length})")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if not self._started:
            if isinstance(frame, StartFrame):
                self._started = True
                logger.debug("LightRAGContextProcessor started")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame) and self.enable_context_injection:
            annotated = await self._annotate(frame)
            await self.push_frame(annotated, direction)
        else:
            await self.push_frame(frame, direction)

    async def _annotate(self, frame: TranscriptionFrame) -> TranscriptionFrame:
        user_query = (frame.text or "").strip()
        if _should_skip(user_query) or self.rag_system is None:
            return frame

        try:
            # Direct call without threading to avoid segfaults
            coro = getattr(self.rag_system, "get_relevant_context", None)
            if asyncio.iscoroutinefunction(coro):
                context = await self.rag_system.get_relevant_context(user_query, max_length=self.max_context_length)
            else:
                context = self.rag_system.get_relevant_context(user_query, self.max_context_length)
            logger.debug(f"LightRAGContextProcessor: got {len(context or '')} chars of context.")
        except Exception as e:
            logger.debug(f"LightRAG context skipped (error): {e}")
            context = ""

        if context and context.strip():
            return RAGAnnotatedTranscriptionFrame(
                text=frame.text,
                user_id=frame.user_id,
                timestamp=frame.timestamp,
                rag_context=context.strip(),
            )
        return frame


# =============================================================================
# Factories
# =============================================================================
def create_rag_context_processor(
    enable_context: bool = True,
    max_context: int = 800,
    enable_calendar: bool = False,
    calendar_context = None,
) -> RAGContextProcessor:
    """Collector only — attaches context to frames (no UI mutation)."""
    return RAGContextProcessor(
        enable_context_injection=enable_context,
        max_context_length=max_context,
        enable_calendar=enable_calendar,
        calendar_context=calendar_context,
    )

def create_rag_prompt_injector_with_context(openai_llm_context, tag: str = "RAG Context") -> RAGPromptInjectorWithContext:
    """
    Injector — appends a SYSTEM message with the context into the provided OpenAILLMContext.
    Place immediately BEFORE context_aggregator.user() in your pipeline.
    """
    return RAGPromptInjectorWithContext(openai_llm_context, tag=tag)

def create_lightrag_context_processor(
    max_context: int = 400,
    enable_context: bool = True,
) -> LightRAGContextProcessor:
    """LightRAG collector (no UI leaks); pair with create_rag_prompt_injector_with_context()."""
    return LightRAGContextProcessor(
        max_context_length=max_context,
        enable_context_injection=enable_context,
    )
