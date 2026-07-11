# Lab 3.2 — RAG Pipeline

CalderR Agentic AI Engineering Internship · Week 3

This lab spans **Tuesday** (ingestion pipeline + vector DBs) and **Wednesday** (RAG query + generation).
This README covers the Tuesday portion.

---

## Tuesday: Vector DBs & Ingestion Pipeline

### The three tools built here

| File | What it does |
|---|---|
| `fetch_docs.py` | Downloads 10 Wikipedia articles → `docs/*.txt` + `docs/*.json` |
| `ingest.py` | Loads docs → splits into chunks → embeds → stores in ChromaDB |
| `query.py` | Queries ChromaDB with semantic search + metadata filters |
| `faiss_demo.py` | Shows IndexFlatL2 vs IndexIVFFlat using Lab 3.1 embeddings |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Fetch documents

```bash
python fetch_docs.py              # downloads 10 Wikipedia articles
python fetch_docs.py list-docs    # verify what was downloaded
```

Downloads articles on: AI, Machine Learning, Deep Learning, NLP, Large Language Models,
Transformers, Vector Databases, RAG, Reinforcement Learning, Computer Vision.

Each article is ~2,000–8,000 words. Together they're ~50 pages of content.

---

## Step 2 — Ingest into ChromaDB

```bash
# Default: chunk_size=512 chars
python ingest.py run

# Experiment with different chunk sizes
python ingest.py run --chunk-size 256
python ingest.py run --chunk-size 1024

# Force re-ingest (wipe and redo)
python ingest.py run --chunk-size 512 --reset

# Inspect what was stored
python ingest.py stats
python ingest.py stats --chunk-size 256
python ingest.py list-collections
```

Each chunk size creates a **separate collection** so you can compare them side-by-side.
This is the chunk size experiment required for Wednesday's evaluation.

**What the metadata looks like per chunk:**
```json
{
  "source": "deep_learning",
  "topic": "Deep learning",
  "url": "https://en.wikipedia.org/wiki/Deep_learning",
  "fetch_date": "2026-07-08",
  "chunk_index": 12,
  "total_chunks": 47,
  "word_count": 84,
  "char_count": 498,
  "chunk_size_config": 512
}
```

---

## Step 3 — Query with filters

```bash
# Basic semantic search
python query.py ask "how do neural networks learn?"

# Filter to a specific document
python query.py ask "attention mechanism" --source transformer_deep_learning_architecture

# Filter by minimum word count (avoid tiny stub chunks)
python query.py ask "training data and loss functions" --min-words 60

# Combined filter
python query.py ask "language model pretraining" \
  --source large_language_model --min-words 50 --top-k 8

# See available document slugs (for --source filter)
python query.py sources

# Inspect ingestion quality — view first 5 chunks of a doc
python query.py peek deep_learning
python query.py peek large_language_model --n 3
```

---

## Step 4 — FAISS comparison

```bash
# Exact search with IndexFlatIP
python faiss_demo.py search --query "convolutional neural networks"

# Compare exact vs approximate side-by-side
python faiss_demo.py search --query "convolutional neural networks" --compare

# Conceptual explanation
python faiss_demo.py explain
```

---

## Core concepts covered

### ChromaDB storage model

```
PersistentClient (./chroma_db/)
└── Collection: wiki_docs_chunk512
    ├── id: "deep_learning_c0012"
    ├── document: "Convolutional neural networks use..."    ← the chunk text
    ├── embedding: [0.23, -0.41, 0.17, ...]               ← 384-dim vector
    └── metadata: {source, chunk_index, word_count, ...}  ← filterable fields
```

### In-memory vs Persistent vs Server

| Mode | Code | Use case |
|---|---|---|
| In-memory | `chromadb.EphemeralClient()` | Testing, notebooks, throwaway |
| Persistent | `chromadb.PersistentClient(path="./db")` | Local dev, single-machine prod |
| Server | `chromadb.HttpClient(host="...", port=8000)` | Multi-service, Docker Compose |

### FAISS index types

| Index | Search type | Speed | Memory | Needs training |
|---|---|---|---|---|
| `IndexFlatL2` / `IndexFlatIP` | Exact | Slow at scale | High | No |
| `IndexIVFFlat` | Approximate | Fast | Medium | Yes (k-means) |
| `IndexHNSW` | Approximate | Very fast | Medium | No |
| `IndexPQ` | Approximate | Fast | Low (compressed) | Yes |

### ChromaDB where-clause syntax

```python
# Single field
where={"source": {"$eq": "deep_learning"}}

# Numeric comparison
where={"word_count": {"$gte": 50}}

# AND condition
where={"$and": [
    {"source": {"$eq": "machine_learning"}},
    {"word_count": {"$gte": 40}}
]}

# OR condition
where={"$or": [
    {"source": {"$eq": "deep_learning"}},
    {"source": {"$eq": "machine_learning"}}
]}
```

---

## Wednesday: RAG Pipeline (coming next)

Wednesday adds `rag.py` and `chunk_eval.py` to this same directory:
- `rag.py` — wires ChromaDB retrieval to LangChain + Groq for answer generation
- `chunk_eval.py` — runs 20 Q&A pairs against chunk sizes 256/512/1024 and reports accuracy

---

## File structure

```
lab-3-2-rag-pipeline/
├── fetch_docs.py        # Step 1: download Wikipedia articles
├── ingest.py            # Step 2: chunk → embed → ChromaDB
├── query.py             # Step 3: semantic search + metadata filters
├── faiss_demo.py        # FAISS comparison
├── rag.py               # Step 4: retrieval + generation (Wednesday)
├── chunk_eval.py        # Chunk size evaluation (Wednesday)
├── requirements.txt
├── README.md
├── docs/                # downloaded articles (gitignore this)
│   ├── deep_learning.txt
│   └── deep_learning.json
└── chroma_db/           # ChromaDB persistence (gitignore this)
```

Add to `.gitignore`:
```
docs/
chroma_db/
```
