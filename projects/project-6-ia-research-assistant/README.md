# Project 6-I-A: Long-Term Personal Research Assistant

A personalized agentic research system built with **Python 3.11+**, **ChatGroq (llama-3.3-70b-versatile)**, **SQLite**, **ChromaDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **Pydantic**, and **Streamlit**. 

The research assistant maintains long-term memory across sessions using a dual-store architecture (lossless SQLite episodic history + ChromaDB vector index) and an explicit **Mem0 Profile Reconciler** that dynamically updates user preferences and known topics using `ADD`, `UPDATE`, `DELETE`, and `NOOP` operations.

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │   User Query & Turns   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Session Initialiser   │
                       └───────────┬────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     ┌─────────────────────┐               ┌────────────────────┐
     │ SQLite Episodic     │               │ ChromaDB Semantic  │
     │ Store (Turn Logs)   │               │ Index (Embeddings) │
     └──────────┬──────────┘               └─────────┬──────────┘
                │                                     │
                └──────────────────┬──────────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ ChatGroq LLM Engine    │ ◄─── Current UserProfile
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Generated Response    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Post-Session Memory    │
                       │ & Profile Reconciler   │
                       └────────────────────────┘
```

### System Components
- **Episodic Store (`episodic_store.py`)**: SQLite database (`episodic_memory.db`) storing exact, lossless conversation turns (`session_id`, `timestamp`, `role`, `content`, `importance_score`).
- **Semantic Store (`semantic_store.py`)**: ChromaDB persistent index calculating composite memory scores.
- **Mem0 Profile Reconciler (`reconciler.py`)**: Reconciles extracted facts against the persistent `UserProfile` using ChatGroq.
- **Research Agent (`agent.py`)**: Generates personalized responses using ChatGroq LLM that adapt length, summarize known topics, and connect new questions to prior research.

---

## 🔄 Profile Reconciliation (Mem0 ADD / UPDATE / DELETE / NOOP)

| Operation | Target Field Type | Example Scenario | Resolution Action |
| :--- | :--- | :--- | :--- |
| **`UPDATE`** | Singleton (e.g., `preferred_depth`) | User states *"I'd prefer brief, high-level answers from now on."* while profile has `"detailed"`. | Overwrites old preference `"detailed"` with `"brief"`. The profile maintains exactly **one** current preference. |
| **`ADD`** | Collection (e.g., `known_topics`) | User asks about a new topic: *"What is multi-head attention?"* | Appends `"multi-head attention"` to `known_topics`. |
| **`NOOP`** | Collection / Singleton | User asks about `"self-attention"` which is already recorded. | Ignores duplicate candidate fact to prevent redundant profile growth. |
| **`DELETE`** | Collection / Singleton | User explicitly retracts a topic or preference. | Removes entry from profile collection. |

---

## 🚀 Usage

### 1. Launch 3-Panel Streamlit Dashboard
Launch the interactive web UI featuring Chat, Memory Inspector, and Profile Viewer:

```bash
uv run streamlit run projects/project-6-ia-research-assistant/dashboard.py
```

### 2. Run Real LLM Smoke Test Suite
Run the verification proofs using ChatGroq:

```bash
uv run python projects/project-6-ia-research-assistant/smoke_test.py
```

---

## 📂 Directory Structure

```text
projects/project-6-ia-research-assistant/
├── README.md               # System architecture & user guide
├── agent.py                # Personalized research agent loop
├── dashboard.py            # 3-panel Streamlit interface
├── episodic_store.py       # SQLite turn history logger
├── profile_store.py        # Persistent JSON UserProfile manager
├── reconciler.py           # Mem0 profile reconciliation engine
├── semantic_store.py       # ChromaDB vector store
└── smoke_test.py           # Verification suite
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Storage:** SQLite & ChromaDB
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Frontend:** Streamlit
