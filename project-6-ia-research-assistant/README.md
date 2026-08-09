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
- **Semantic Store (`semantic_store.py`)**: ChromaDB persistent index calculating composite memory scores:
  $$\text{Score} = \text{MinMax}(\text{Recency}) + \text{MinMax}(\text{Relevance})$$
  where recency is exponential decay ($0.995^{\text{hours}}$) and relevance is $384$-dimensional cosine similarity.
- **Mem0 Profile Reconciler (`reconciler.py`)**: Reconciles extracted facts against the persistent `UserProfile` using ChatGroq.
- **Research Agent (`agent.py`)**: Generates personalized responses using ChatGroq LLM that adapt length, summarize known topics, and connect new questions to prior research.

---

## 🔄 Profile Reconciliation (Mem0 ADD / UPDATE / DELETE / NOOP)

Traditional chatbots blindly append chat turns into history, causing preference conflicts (e.g. storing both `"brief"` and `"detailed"` depth simultaneously) and context window bloat. This project implements Mem0's structured reconciliation protocol:

| Operation | Target Field Type | Example Scenario | Resolution Action |
| :--- | :--- | :--- | :--- |
| **`UPDATE`** | Singleton (e.g., `preferred_depth`) | User states *"I'd prefer brief, high-level answers from now on."* while profile has `"detailed"`. | Overwrites old preference `"detailed"` with `"brief"`. The profile maintains exactly **one** current preference. |
| **`ADD`** | Collection (e.g., `known_topics`) | User asks about a new topic: *"What is multi-head attention?"* | Appends `"multi-head attention"` to `known_topics`. |
| **`NOOP`** | Collection / Singleton | User asks about `"self-attention"` which is already recorded. | Ignores duplicate candidate fact to prevent redundant profile growth. |
| **`DELETE`** | Collection / Singleton | User explicitly retracts a topic or preference. | Removes entry from profile collection. |

Each decision logs written technical reasoning (e.g. `reasoning="Updated preferred_depth from 'detailed' to 'brief' based on new user preference."`).

---

## ⚡ Personalization & Behavior Adaptation Proof

Below is a side-by-side comparison of the agent's behavior evolution across the 5-session scenario:

| Session | User Query & Context | Assistant LLM Response | Behavioral Adaptation Proven |
| :--- | :--- | :--- | :--- |
| **Session 1** | *"Explain self-attention in transformers to me in detail"*<br>*(Profile: `preferred_depth="detailed"`)* | *"Self-attention is a fundamental component of transformer architectures... Input sequence is linearly transformed into Q, K, V matrices. Attention scores are computed via dot product: softmax(Q * K^T / sqrt(d))..."* | **Full Technical Detail**: Generates an exhaustive explanation (~450 words) matching initial `detailed` preference. |
| **Session 4** | *"Remind me what self-attention is."*<br>*(Profile: `preferred_depth="brief"`, `self-attention` in `known_topics`)* | *"This topic was previously covered in earlier research. Self-attention is a mechanism allowing models to attend to different parts of input sequence..."* | **Length Adaptation & Prior Coverage Acknowledgment**: Response is **measurably shorter** (~70 words vs 450 words) and explicitly references prior coverage (*"previously covered in earlier research"*). |
| **Session 5** | *"How does positional encoding relate to what I asked about earlier?"*<br>*(Cross-topic synthesis)* | *"Positional encoding is a technique used in transformers... It relates to **self-attention**, as it helps the **self-attention** mechanism understand sequence order... This is also relevant to **multi-head attention**..."* | **Proactive Multi-Topic Connection**: Synthesizes the current query by explicitly naming prior topics (`self-attention` and `multi-head attention`). |

---

## 🚀 Usage

### 1. Launch 3-Panel Streamlit Dashboard
Launch the interactive web UI featuring Chat, Memory Inspector, and Profile Viewer:

```bash
uv run streamlit run project-6-ia-research-assistant/dashboard.py
```

### 2. Run Real LLM Smoke Test Suite
Run the 5 real LLM verification proofs using ChatGroq:

```bash
uv run python project-6-ia-research-assistant/smoke_test.py
```

---

## 🔑 Environment Setup

Ensure `GROQ_API_KEY` is set in your `.env` file at the root of the repository:

```env
GROQ_API_KEY=gsk_...
```
