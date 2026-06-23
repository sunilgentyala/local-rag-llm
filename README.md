# 🧠 Local RAG LLM

> **A privacy-first, fully local Retrieval-Augmented Generation (RAG) application.**
> Chat with your own PDF documents — no internet required, no API keys, no data leaves your machine.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-green)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-local-orange)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://github.com/sunilgentyala/local-rag-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/sunilgentyala/local-rag-llm/actions/workflows/ci.yml)

**🌐 [Project website](https://sunilgentyala.github.io/local-rag-llm/)**

---

## 📌 Project Overview

This application lets you upload any number of PDF documents and ask natural-language questions about their contents. All inference runs locally on your CPU using [Ollama](https://ollama.com) — there is no external API, no cloud dependency, and no cost per query.

| Property | Detail |
|---|---|
| **LLM** | `llama3` (8B, Q4) via Ollama |
| **Embeddings** | `nomic-embed-text` via Ollama |
| **Framework** | LangChain 1.x (LCEL) |
| **Vector Store** | ChromaDB (persistent, local) |
| **UI** | Streamlit |
| **Privacy** | 100% offline · no telemetry |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    Streamlit (app.py)                           │
│          PDF Upload │ Chat Input │ Response Display             │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────────────────┐
    │   INGESTION PIPELINE │    │       RAG QUERY PIPELINE        │
    │                      │    │                                  │
    │  PyPDFLoader         │    │  User Question                   │
    │       ↓              │    │       ↓                          │
    │  RecursiveText       │    │  OllamaEmbeddings                │
    │  Splitter            │    │  (nomic-embed-text)              │
    │       ↓              │    │       ↓                          │
    │  OllamaEmbeddings    │    │  ChromaDB Similarity Search      │
    │  (nomic-embed-text)  │    │  (Top-K chunks retrieved)        │
    │       ↓              │    │       ↓                          │
    │  ChromaDB            │    │  Prompt Assembly                 │
    │  (persist to disk)   │    │  (System + Context + Question)   │
    └──────────────────────┘    │       ↓                          │
                                │  ChatOllama (llama3)             │
                                │       ↓                          │
                                │  Answer + Source Citations       │
                                └──────────────────────────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │   LOCAL STORAGE ONLY    │
                               │   .chroma/  (vectors)   │
                               │   No cloud. No API key. │
                               └─────────────────────────┘
```

---

## ✨ Features

- **Multi-PDF Upload** — ingest one or many documents in a single session.
- **Persistent Vector Store** — ChromaDB is saved to disk; restart the app without re-embedding.
- **Source Citations** — every answer links back to the exact source file and page number.
- **Clear & Reset** — one-click button to wipe the database and start fresh.
- **Hallucination guard** — system prompt instructs the model to stay strictly within provided context.
- **LCEL chain** — built on LangChain Expression Language for full compatibility with LangChain 1.x.

---

## ⚡ Quick Start

### 1 — Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download installer from https://ollama.com/download
```

### 2 — Pull the required models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 3 — Clone the repo

```bash
git clone https://github.com/sunilgentyala/local-rag-llm.git
cd local-rag-llm
```

### 4 — Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5 — Start Ollama (in a separate terminal)

```bash
ollama serve
```

### 6 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Repository Structure

```
local-rag-llm/
├── app.py                  # Main Streamlit application
├── db.py                   # SQLite chat history
├── generators.py           # IEEE / Elsevier / Blog DOCX generation
├── requirements.txt        # Python dependencies (LangChain 1.x)
├── Dockerfile              # Container build for the app
├── docker-compose.yml      # App + Ollama, one command
├── .env.example            # Optional Ollama endpoint / model overrides
├── .gitignore              # Ignores venv, .chroma, uploads
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # Contribution guidelines
├── README.md               # This file
├── .github/workflows/ci.yml# Lint + compile check on push/PR
└── docs/
    ├── index.html          # GitHub Pages project site
    ├── ABSTRACT.md         # Formal project abstract
    ├── RESEARCH.md         # RAG vs Fine-Tuning analysis + future roadmap
    └── TESTING_QA.md       # Testing matrix and QA test cases
```

---

## 🗺️ Roadmap

- [x] Multi-PDF RAG with local LLM
- [x] Persistent ChromaDB storage
- [x] Source citation display
- [x] LangChain 1.x LCEL compatibility
- [ ] DOCX / TXT / CSV document support
- [ ] Conversation memory (multi-turn context)
- [ ] QLoRA fine-tuning notebook (Google Colab + Unsloth)
- [ ] Quantized model benchmarks (Q4 vs Q8)

---

## 🔒 Privacy Guarantee

This application is designed with a **privacy-first** architecture:

- No data is transmitted to any external server.
- No API key is required.
- All embeddings and inference run on-device via Ollama.
- ChromaDB stores vectors locally in the `.chroma/` directory (gitignored).
- Uploaded PDFs are processed in memory via `tempfile` and immediately deleted.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
Reach me at sunil.gentyala@ieee.org

---

> Built with LangChain · Ollama · ChromaDB · Streamlit
