# 🛰️ c-beam AI — The C-Base Station Computer

c-beam is the sarcastic, occasionally grumpy, but always *reluctantly helpful* AI system running inside the crashed **c-base space station**. Think of it as HAL 9000’s laid-back Berlin cousin — fueled by sarcasm, caffeine, and Python.

This project turns c-beam into a fully modular station computer with real conversational abilities, knowledge retrieval, and database querying. It’s built to be extensible, hackable, and fun to tinker with.

---

## ✨ Features

* **Conversational Engine (PipeCat)**
  Handles speech-to-text, text-to-speech, and the overall chat pipeline. Keeps the sarcasm flowing.

* **Vector Retrieval (VectorRAG)**
  For pulling knowledge from station archives, manuals, and “ancient” PDFs. Great for answering lore questions like *“How many rings does the c-base have?”*

* **Text-to-SQL Agent**
  Connects to event databases and the station calendar. Lets you ask:

  > *“Hey c-beam, what’s happening in the main hall next Friday?”*
  > …and it’ll query the DB for you.

* **Extensible Architecture**
  Swap out models, add new agents, or plug in other APIs. The system is designed for hackerspaces and AI enthusiasts who like to break things until they work.

---

## 🚀 Quick Start

1. Clone the repo:

   ```bash
   git clone https://github.com/SasquatchPDX/cbeam_pipecat.git
   cd c-beam_pipecat
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)

   pip install -r requirements.txt
   ```

3. Configure your `.env` file with API keys and DB credentials. Example:

   ```env
   OPENAI_API_KEY=your-key-here
   DATABASE_URL=sqlite:///cbase_events.db
   ```

4. Fire up the station computer:

   ```bash
   python bot.py
   ```

---

## 🛠️ Tech Stack

* [PipeCat](https://github.com/pipecat-ai) – Conversational pipelines (STT, TTS, chat orchestration)
* Local/remote LLMs via Ollama, OpenAI, or other backends
* Vector DB (Chroma, SQLite, or your choice) for RAG
* DuckDB / SQLAlchemy for text-to-SQL event queries

---

## 🤖 Personality

c-beam isn’t just another chatbot — it’s the station’s reluctant roommate:

* **Tone**: sarcastic, witty, slightly grumpy
* **Style**: short banter before giving useful info
* **Goal**: always provide answers… eventually

---
Architecture

graph TB
      subgraph "Client Layer (React/TypeScript)"
          UI[Terminal UI Interface]
          WRT[WebRTC Transport]
          MSG[MessagesPanel - Real-time Transcripts]
          LLM_SEL[LLM Provider Selector]
          LANG[Language Context]
          BOOT[Boot Sequence Animation]
      end

      subgraph "Voice Processing Pipeline (Pipecat)"
          STT[WhisperSTT Service<br/>Speech-to-Text]
          TTS[KokoroTTS/NativeTTS<br/>Text-to-Speech]
          VAD[SileroVAD<br/>Voice Activity Detection]
          TRANS[Daily WebRTC Transport<br/>Port 7860]
      end

      subgraph "Core Processing Chain"
          IMM[ImmediateTranscriptProcessor<br/>Instant UI Updates]
          CONV[ConversationLogger<br/>SQLite Storage]
          TXT2SQL[Txt2SQLEnricher<br/>Calendar Queries]
          RAG[RAGContextProcessor<br/>Knowledge Search]
          ML[MultilingualProcessor<br/>Translation Support]
          CONTEXT[OpenAI LLM Context<br/>Message Threading]
      end

      subgraph "Knowledge Systems"
          subgraph "RAG System"
              RAG_DB[(Vector Database<br/>957 Chunks)]
              KB[Knowledge Base<br/>German/English Docs]
              EMBED[Ollama Embeddings<br/>nomic-embed-text]
          end

          subgraph "Calendar System"
              CAL_DB[(DuckDB Calendar<br/>31,243 Events)]
              ICAL[c-base iCal Feed<br/>Auto-sync]
              SQL_GEN[Natural Language → SQL<br/>duckdb-nsql:7b]
          end
      end

      subgraph "AI/LLM Layer"
          OLLAMA[Ollama Server<br/>Port 11434]
          GPT_OSS[gpt-oss:20b Model<br/>Chat Completions]
          DUCKDB_MODEL[duckdb-nsql:7b-q2_K<br/>SQL Generation]
          EMBED_MODEL[nomic-embed-text<br/>Embeddings]
      end

      subgraph "API Services"
          API_SRV[API Server<br/>Port 5001]
          SWITCH[LLM Provider Switching<br/>Ollama/OpenAI/GPT5]
      end

      subgraph "Character & Personality"
          CASS[Cassandra AI<br/>Grumpy but Helpful<br/>c-base Station AI]
          SYS_INST[System Instructions<br/>Personality & Behavior]
          RESPONSES[Canned Responses<br/>Immediate Feedback]
      end

      subgraph "Data Storage"
          CONV_DB[(conversations.db<br/>SQLite)]
          VECTOR_DB[(knowledge_vectors.json<br/>Nano-VectorDB)]
          META_DB[(knowledge_metadata.json<br/>Document Metadata)]
      end

      %% User Flow
      USER[👤 User] --> UI
      UI <--> WRT
      WRT <--> TRANS

      %% Voice Pipeline
      TRANS --> VAD
      VAD --> STT
      STT --> IMM

      %% Processing Chain
      IMM --> MSG
      IMM --> CONV
      CONV --> TXT2SQL
      TXT2SQL --> RAG
      RAG --> ML
      ML --> CONTEXT

      %% LLM Processing
      CONTEXT --> OLLAMA
      OLLAMA --> GPT_OSS
      GPT_OSS --> CONTEXT
      CONTEXT --> TTS
      TTS --> TRANS

      %% Knowledge Systems
      TXT2SQL <--> CAL_DB
      TXT2SQL <--> SQL_GEN
      SQL_GEN <--> DUCKDB_MODEL
      CAL_DB <--> ICAL

      RAG <--> RAG_DB
      RAG_DB <--> KB
      RAG_DB <--> EMBED_MODEL
      EMBED_MODEL <--> OLLAMA

      %% API Layer
      UI <--> API_SRV
      API_SRV <--> SWITCH
      SWITCH <--> OLLAMA

      %% Character Integration
      CONTEXT --> CASS
      CASS --> SYS_INST
      TXT2SQL --> RESPONSES

      %% Data Storage
      CONV --> CONV_DB
      RAG --> VECTOR_DB
      RAG --> META_DB

      %% Styling
      classDef userInterface fill:#e1f5fe,stroke:#01579b,stroke-width:2px
      classDef voiceProcessing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
      classDef coreProcessing fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
      classDef knowledge fill:#fff3e0,stroke:#e65100,stroke-width:2px
      classDef aiLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
      classDef storage fill:#f1f8e9,stroke:#33691e,stroke-width:2px
      classDef character fill:#ede7f6,stroke:#311b92,stroke-width:2px

      class UI,WRT,MSG,LLM_SEL,LANG,BOOT userInterface
      class STT,TTS,VAD,TRANS voiceProcessing
      class IMM,CONV,TXT2SQL,RAG,ML,CONTEXT coreProcessing
      class RAG_DB,KB,EMBED,CAL_DB,ICAL,SQL_GEN knowledge
      class OLLAMA,GPT_OSS,DUCKDB_MODEL,EMBED_MODEL,API_SRV,SWITCH aiLayer
      class CONV_DB,VECTOR_DB,META_DB storage
      class CASS,SYS_INST,RESPONSES character


---

## 🧑‍🚀 Contributing

Pull requests, feature suggestions, and new agents are welcome.
If you break the station’s life support with a PR, we’ll just blame you in the commit history.

---

## 📜 License

MIT… or maybe something more “don’t-sell-this-without-buying-us-a-beer.” TBD.

