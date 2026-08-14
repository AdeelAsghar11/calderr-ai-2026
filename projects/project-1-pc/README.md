# Project 1-P-C: Agentic Research Assistant

An autonomous multi-step research assistant built with **Streamlit**, **LangChain**, and **Groq (`llama-3.3-70b-versatile`)**.

The assistant dynamically plans subtopics, executes sequential investigation loops over each area with structured confidence scoring, and synthesizes an executive markdown report with interactive confidence analytics.

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │  User Query Input      │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Dynamic Planner Agent  │
                       │ (3-5 Subtopics Plan)   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Sequential Research    │
                       │ (Subtopic Fact Finder) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Synthesis Engine       │
                       │ (Confidence Aggregation)│
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Streamlit Dashboard UI │
                       └────────────────────────┘
```

---

## ✨ Features

- **Dynamic Subtopic Planning:** Breaks complex queries into 3–5 distinct subtopics for targeted analysis.
- **Sequential Exploration Loop:** Evaluates subtopics individually, extracting findings and calculating confidence ratings.
- **Automated Synthesis Engine:** Assembles findings into an executive report detailing core findings, limitations, and overall confidence score.
- **Streamlit Interactive UI:**
  - Real-time step progress updates.
  - Subtopic confidence level comparisons.
  - One-click markdown report exporter.

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### Run Streamlit App
From the repository root:
```bash
uv run streamlit run projects/project-1-pc/research_assistant.py
```

---

## 📂 Directory Structure

```text
projects/project-1-pc/
├── README.md               # Project architecture & user guide
└── research_assistant.py   # Streamlit app & research pipeline
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend UI:** Streamlit
- **Framework:** LangChain & LangChain-Groq
