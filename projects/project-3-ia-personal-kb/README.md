# Project 3-I-A: Personal Knowledge Base

A Retrieval-Augmented Generation (RAG) system built with **Python 3.11+**, **ChromaDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **LangChain**, and **Groq (`llama-3.3-70b-versatile`)**.

The system ingests multi-format documents (PDF, DOCX, Markdown, TXT), generates semantic vector embeddings, stores them in a local ChromaDB collection, and provides grounded Q&A with source citations.

---

## 🏗️ Architecture

```
                        INGESTION (run once)
┌──────────┐    ┌───────────┐    ┌───────────────┐    ┌───────────┐
│  docs/   │───▶│  loader   │───▶│  text splitter │───▶│ embedder  │
│ .pdf/.md │    │ (multi-   │    │  (512 char,    │    │(MiniLM-L6)│
│ .txt/    │    │  format)  │    │   50 overlap)  │    │           │
│ .docx    │    └───────────┘    └───────────────┘    └─────┬─────┘
└──────────┘                                                 │
                                                               ▼
                                                        ┌─────────────┐
                                                        │  ChromaDB   │
                                                        │ personal_kb │
                                                        └─────────────┘

                          QUERY (every question)
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌─────────────┐
│ question │───▶│  embed    │───▶│ vector search │───▶│  context    │
└──────────┘    │  (same    │    │ (+ optional   │    │  assembly   │
                 │   model)  │    │  source       │    │ (numbered,  │
                 └───────────┘    │  filter)      │    │  cited)     │
                                   └──────────────┘    └──────┬──────┘
                                                              │
                                                              ▼
                                                        ┌──────────────┐
                                                        │  Groq LLM    │
                                                        │ (streaming)  │
                                                        └──────┬───────┘
                                                              │
                                                              ▼
                                                        answer + sources
```

---

## ✨ Features

- **Multi-Format Document Ingestion:** Built-in loader supporting PDF (`pypdf`), DOCX (`python-docx`), Markdown, and plain text.
- **Persistent Vector Store:** Local ChromaDB instance with collection metadata management.
- **Filtered Semantic Retrieval:** Supports full collection search or document-level source filtering.
- **Grounded Q&A & Citations:** Explicit system prompt constraints enforcing answer grounding and exact source chunk attribution.
- **Interactive CLI & Chat:** `ask`, `chat`, and `sources` commands.

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### Ingestion & CLI Commands
Run commands from the project directory `projects/project-3-ia-personal-kb` or repository root:

```bash
# Ingest all files in docs/
uv run python projects/project-3-ia-personal-kb/ingest.py run

# View index statistics
uv run python projects/project-3-ia-personal-kb/ingest.py stats

# Ask a single question
uv run python projects/project-3-ia-personal-kb/kb.py ask "What projects are in my portfolio?" --show-context

# Interactive Chat Session
uv run python projects/project-3-ia-personal-kb/kb.py chat
```

---

## 📂 Directory Structure

```text
projects/project-3-ia-personal-kb/
├── README.md               # Architecture & documentation
├── docs/                   # Multi-format source document directory
├── generate_qa_examples.py # Benchmark report generator
├── ingest.py               # Document ingestion & embedding runner
├── kb.py                   # Main CLI interface (ask, chat, sources)
├── kb_chroma_db/           # ChromaDB local persistence directory
├── loader.py               # Document loader (PDF, DOCX, MD, TXT)
└── requirements.txt        # Dependencies
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Vector Database:** ChromaDB
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Framework:** LangChain & LangChain-Groq
