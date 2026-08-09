# GraphRAG vs Vector Search: Proving When Hybrid Retrieval Is Actually Required

*A deep dive into dual indexing with NetworkX and ChromaDB, pre-retrieval query routing, and statistical significance testing.*

---

## Introduction

Retrieval-Augmented Generation (RAG) systems have become the standard architecture for grounding Large Language Models on domain-specific knowledge. However, standard vector search frequently fails on multi-hop relational questions (such as *"Who founded the company that employee X works at?"*), while pure Knowledge Graph search fails on unstructured descriptive attributes (such as *"What is founder X's prior profession?"*).

In this technical post, we present **Project 6-P-B GraphRAG Knowledge Intelligence**, a dual-indexed retrieval architecture that combines ChromaDB vector search, NetworkX graph neighborhood expansion, pre-retrieval query routing, and RAGAS metric evaluation.

---

## 🏗️ Architecture & Dual Indexing

The system maintains two parallel indices over the same underlying domain corpus:

```text
                                 ┌────────────────────────┐
                                 │   User Query String    │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │   Pre-Retrieval Router │
                                 └───────────┬────────────┘
                                             │
           ┌─────────────────────────────────┼─────────────────────────────────┐
           ▼                                 ▼                                 ▼
    (Factual Route)                  (Relational Route)                 (Complex Route)
  ┌───────────────────┐             ┌──────────────────┐             ┌──────────────────┐
  │ Vector Search     │             │ Graph Expansion  │             │  Hybrid Fusion   │
  │ (ChromaDB L2)     │             │ (NetworkX 2-Hop) │             │ (Vector + Graph) │
  └─────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
            │                                │                                │
            └────────────────────────────────┼────────────────────────────────┘
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

1. **ChromaDB Vector Index**: Encodes 55 corpus documents into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
2. **NetworkX Knowledge Graph**: Maintains a directed graph of 25 unique entity nodes and 27 directed relation edges (`founded_by`, `located_in`, `part_of`, `works_at`).

---

## 🧪 True-By-Construction Corpus Methodology

To evaluate GraphRAG rigorously without relying on lucky co-occurrences, the 55-paragraph corpus enforces a **structural gap**:
- **Separate Documents**: Multi-hop relational chains are strictly separated across distinct documents. For example, document A states `Farah Deng works at Ridgeline Dynamics`, while document B states `Marcus Ondiek founded Ridgeline Dynamics`.
- **Descriptive Facts**: Background facts (e.g. `Marcus Ondiek worked as a renewable energy engineer`) contain zero graph relationship edges.

Under this design:
- **Vector-only** search easily retrieves static descriptive facts but fails to trace 2-hop relational paths.
- **Graph-only** search traces relational paths but misses unstructured descriptive facts.
- **Hybrid retrieval** is provably required to answer complex multi-hop descriptive questions.

---

## 📊 Evaluation Results (Pending Real LLM Quota Reset)

> **Evaluation Status Notice:**
> The statistical test (t=11.33, p<0.00001) demonstrated so far used synthetic stub scores to prove the statistical test machinery itself works correctly, not real RAGAS-scored data — a real evaluation run is currently pending due to a Groq API rate limit hit during Phase 2, to be completed separately once quota resets.

### Placeholder Results Table (Pending Real Execution)

| Question Category | Vector-Only Mean | Graph-Only Mean | Hybrid Mean | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **Factual** | *[Pending]* | *[Pending]* | *[Pending]* | N/A |
| **Relational** | *[Pending]* | *[Pending]* | *[Pending]* | N/A |
| **Complex** | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending (Paired t-test)]* |

---

## 🚀 Conclusion & Next Steps

Phase 3 completes the core backend architecture, FastAPI control plane, Streamlit research interface, and containerized Docker setup. Once the Groq API daily quota resets, the complete 90-sample real RAGAS evaluation will be executed and published in this post.
