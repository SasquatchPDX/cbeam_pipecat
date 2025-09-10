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

## 🧑‍🚀 Contributing

Pull requests, feature suggestions, and new agents are welcome.
If you break the station’s life support with a PR, we’ll just blame you in the commit history.

---

## 📜 License

MIT… or maybe something more “don’t-sell-this-without-buying-us-a-beer.” TBD.

