# Project 3-P-B: Production Research Engine

An enterprise-grade, portfolio-level RAG Research System built with **FastAPI**, **Streamlit**, **ChromaDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **BM25**, and **Groq (`llama-3.3-70b-versatile`)**.

The engine implements pre-retrieval query routing, dual hybrid vector + BM25 search, Reciprocal Rank Fusion (RRF), structured multi-section report generation, persistent JSON storage, and a dual access model (FastAPI REST API control plane + Streamlit interactive UI).

---

## 🏗️ Architecture

```
                               ┌────────────────────────┐
                               │   User Query Input     │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  Pre-Retrieval Router  │
                               │  (KnowledgeRouter)     │
                               └───────────┬────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────────┐
               ▼                           ▼                           ▼
        (Vector Search)            (Hybrid RRF Search)          (Web Search / Direct)
     ┌──────────────────┐        ┌────────────────────┐       ┌─────────────────────┐
     │ ChromaDB Dense   │        │ ChromaDB + BM25    │       │ Fallback Retrieval  │
     └─────────┬────────┘        └─────────┬──────────┘       └──────────┬──────────┘
               │                           │                             │
               └───────────────────────────┼─────────────────────────────┘
                                           ▼
                               ┌────────────────────────┐
                               │ RRF Rank Fusion &      │
                               │ Context Assembler      │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Report Generator Engine│
                               │ (ReportGenerator)      │
                               └───────────┬────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       ┌────────────────────────┐                    ┌────────────────────────┐
       │ FastAPI REST API       │                    │ Streamlit Dashboard    │
       │ (port 8000)            │                    │ (app.py)               │
       └────────────────────────┘                    └────────────────────────┘
```

---

## ✨ Features

- **Pre-Retrieval Query Router:** Classifies queries to determine optimal retrieval strategy (Vector-only, BM25, Hybrid RRF).
- **Dual Hybrid Retrieval:** Merges dense vector embeddings (`all-MiniLM-L6-v2`) with sparse keyword matching (`BM25Okapi`) via Reciprocal Rank Fusion.
- **Structured Multi-Section Report Generation:** Synthesizes long-form markdown research reports complete with executive summaries, detailed analysis, and numbered source citations.
- **FastAPI Control Plane:** Asynchronous REST endpoints (`POST /research`, `GET /reports`, `GET /reports/{id}`, `GET /health`) with OpenAPI documentation (`/docs`).
- **Streamlit Interactive UI:** Search bar, real-time routing decision badge, report viewer with citation explorer, and sidebar report history.
- **Automatic Report Storage:** Saves all generated research reports into structured JSON files in `reports/`.

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### Option A: Launch Streamlit Dashboard
From the repository root:
```bash
uv run streamlit run projects/project-3-pb-research-engine/app.py
```

### Option B: Launch FastAPI REST Server
From the repository root:
```bash
uv run python projects/project-3-pb-research-engine/api.py
```
*API interactive documentation will be available at `http://localhost:8000/docs`.*

---

## 📂 Directory Structure

```text
projects/project-3-pb-research-engine/
├── README.md               # System architecture & documentation
├── api.py                  # FastAPI REST control plane
├── app.py                  # Streamlit interactive frontend
├── generate_demos.py       # Benchmark & report generation script
├── report_generator.py     # Multi-section synthesis engine
├── retriever.py            # Dual hybrid retriever (ChromaDB + BM25 + RRF)
├── router.py               # Pre-retrieval query classifier
├── requirements.txt        # Python dependencies
└── reports/                # Saved JSON research report storage
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Backend API:** FastAPI & Uvicorn
- **Frontend UI:** Streamlit
- **Vector Search & Sparse Retrieval:** ChromaDB, SentenceTransformers (`all-MiniLM-L6-v2`), rank-bm25
- **Data Validation:** Pydantic v2
