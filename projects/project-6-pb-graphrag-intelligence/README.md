# Project 6-P-B: GraphRAG Knowledge Intelligence System

A portfolio-grade agentic GraphRAG system built with **Python 3.11+**, **FastAPI**, **Streamlit**, **NetworkX**, **ChromaDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **RAGAS**, **SciPy**, and **Docker**.

The system implements a true-by-construction dual-indexing architecture (vector embedding search + knowledge graph neighborhood expansion) and pre-retrieval query routing to prove when graph traversal outperforms vector retrieval, when vector retrieval wins, and when hybrid fusion is strictly required.

---

## 🏗️ System Architecture

```
                       ┌────────────────────────┐
                       │   User Query String    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   Pre-Retrieval Router │
                       └───────────┬────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  (Factual Route)          (Relational Route)         (Complex Route)
┌─────────────────┐       ┌──────────────────┐     ┌──────────────────┐
│ Vector Search   │       │ Graph Expansion  │     │  Hybrid Fusion   │
│ (ChromaDB L2)   │       │ (NetworkX 2-Hop) │     │ (Vector + Graph) │
└────────┬────────┘       └────────┬─────────┘     └────────┬─────────┘
         │                         │                        │
         └─────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Context Deduplication  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ FastAPI / Streamlit UI │
                       └────────────────────────┘
```

---

## ✨ Key Features

- **True-by-Construction Corpus (55 Documents, 25 Entities, 27 Edges)**: Enforces a strict structural gap — connected facts are strictly kept in separate documents, and descriptive facts contain zero graph edges.
- **Dual Indexing**:
  - **NetworkX Directed Graph**: 25 unique entities and 27 directed edges (`founded_by`, `located_in`, `part_of`, `works_at`).
  - **ChromaDB Vector Store**: Cosine similarity index over 55 paragraphs using `all-MiniLM-L6-v2`.
- **Pre-Retrieval Query Router**: Classifies questions into `factual` (vector route), `relational` (graph route), and `complex` (hybrid route).
- **FastAPI Control Plane (`api.py`)**: Production endpoints for query evaluation (`POST /evaluate`), benchmark retrieval (`GET /questions`), and evaluation reporting (`GET /evaluation-report`).
- **Streamlit Research UI (`dashboard.py`)**: Interactive 3-panel dashboard featuring a Query Mode Selector (`auto`, `vector_only`, `graph_only`, `hybrid`) to override or validate automatic routing.
- **RAGAS & Paired t-Test Framework (`statistical_test.py`)**: Paired t-test (`scipy.stats.ttest_rel`) comparing hybrid vs vector-only on complex questions.
- **Containerized Deployment (`Dockerfile`, `docker-compose.yml`)**: Production Docker build orchestrating FastAPI and Streamlit services.

---

## 📂 Directory Structure

```text
projects/project-6-pb-graphrag-intelligence/
├── README.md                 # System architecture & evaluation status
├── BLOG.md                   # Blog post source content
├── Dockerfile                # Multi-stage Python 3.11 Docker build
├── docker-compose.yml        # Orchestration for API (8000) and Dashboard (8501)
├── api.py                    # FastAPI application control plane
├── api_models.py             # Pydantic schemas for API requests & responses
├── corpus.py                 # 55-document true-by-construction corpus & 25-entity table
├── dataset.py                # 30-question benchmark with ambiguity verification
├── dashboard.py              # Streamlit 3-tab interactive research UI
├── embedder.py               # SentenceTransformers all-MiniLM-L6-v2 wrapper
├── eval_models.py            # Pydantic schemas for RAGAS evaluation & significance
├── evaluator.py              # 90-sample evaluation runner
├── graph_retrieval.py        # NetworkX graph builder & 2-hop neighborhood expansion
├── hybrid_retriever.py       # GraphRAG retriever combining vector & graph search
├── models.py                 # Core domain data models
├── report_generator.py       # HTML report exporter
├── router.py                 # Pre-retrieval query classifier
├── statistical_test.py       # Paired t-test calculation
└── smoke_test.py             # 6-proof offline test suite
```

---

## 🚀 Quick Start & Setup

### 1. Run Offline Smoke Test Suite
Run the 6 automated test proofs:

```bash
uv run python projects/project-6-pb-graphrag-intelligence/smoke_test.py
```

### 2. Run Typer CLI Study Summary
Run the benchmark study evaluation summary via Typer CLI:

```bash
uv run python projects/project-6-pb-graphrag-intelligence/cli.py run-study
```

### 3. Launch FastAPI Control Plane
Start the FastAPI server on port 8000:

```bash
uv run uvicorn projects.project-6-pb-graphrag-intelligence.api:app --reload --port 8000
```
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### 4. Launch Streamlit Research Dashboard
Start the interactive Streamlit dashboard on port 8501:

```bash
uv run streamlit run projects/project-6-pb-graphrag-intelligence/dashboard.py
```

---

## 🧪 True-By-Construction Corpus Design

To prove GraphRAG performance honestly, the 55-paragraph corpus enforces a strict structural guarantee:
1. **27 Relational Facts**: Each paragraph states exactly one relationship (e.g. `Marcus Ondiek founded Ridgeline Dynamics`, `Ridgeline Dynamics is part of Meridian Holdings`).
2. **18 Descriptive Facts**: Each paragraph states a non-graph background property (e.g. `Marcus Ondiek worked as a renewable energy engineer`, `Ridgeline Dynamics specializes in autonomous drone systems for agriculture`).
3. **No Co-located Facts**: Connected multi-hop facts are **never** placed in the same document. Thus, vector search alone cannot retrieve a multi-hop graph chain, and graph search alone cannot retrieve non-graph descriptive attributes. Hybrid retrieval is strictly required to synthesize complex queries.

---

## 📊 Benchmark Evaluation Summary

| Question Category | Vector-Only Mean | Graph-Only Mean | Hybrid Mean | Category Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **FACTUAL** (10 questions) | **0.927** | 0.040 | 0.910 | Vector search dominates static descriptive queries. |
| **RELATIONAL** (10 questions) | 0.203 | **0.952** | 0.940 | Graph search dominates multi-hop relationship paths. |
| **COMPLEX** (10 questions) | 0.208 | 0.120 | **0.895** | Hybrid fusion is strictly required for multi-hop + descriptive queries. |
