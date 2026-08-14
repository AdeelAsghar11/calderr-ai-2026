# Project 2-I-B: Automated Code Review Agent

An automated multi-pass static code review agent built with **Python 3.11+**, **LangChain**, **Pydantic**, **Groq (`llama-3.3-70b-versatile`)**, and **Rich**. 

The system performs 4 specialized, parallel analysis passes (Bugs, Security, Style, Performance) and synthesizes findings into a unified code review report with severity metrics and line-by-line recommendations.

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │   Source Code Input    │
                       └───────────┬────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼                         ▼
  (Pass 1: Bugs)           (Pass 2: Security)         (Pass 3: Style)          (Pass 4: Performance)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌───────────────────┐
│ Logic Errors    │       │ OWASP Top 10    │       │ PEP 8 & Naming  │       │ Algorithmic Complexity│
│ Null Checks     │       │ Injection & Keys│       │ Docstrings      │       │ Memory & Caching  │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘       └─────────┬─────────┘
         │                         │                         │                           │
         └─────────────────────────┼─────────────────────────┘                           │
                                   ▼                                                     │
                       ┌────────────────────────┐                                        │
                       │ Synthesis Agent        │◄───────────────────────────────────────┘
                       │ (Verdict & Top Pick)   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Rich Terminal Report  │
                       └────────────────────────┘
```

---

## ✨ Features

- **4 Specialized Review Passes:**
  - `bugs`: Detects logic errors, unhandled exceptions, type mismatches, and boundary edge-cases.
  - `security`: Scans for command/SQL injection, hardcoded secrets, unsafe deserialization, and path traversal.
  - `style`: Enforces PEP 8 compliance, docstrings, variable naming, and readability.
  - `performance`: Pinpoints O(n²) bottlenecks, memory leaks, and redundant calculations.
- **Pydantic Structured Output:** Enforces strict typed JSON responses (`ReviewPass`, `Issue`, `CodeReviewReport`).
- **Synthesis Engine:** Calculates category scores (1–10), overall code health score, and actionable final verdict (`approve`, `approve_with_changes`, `request_changes`, `reject`).
- **Rich Terminal Display:** Renders color-coded syntax blocks, severity badges, and category summary tables.

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### Run Code Reviewer
From the repository root:
```bash
uv run python projects/project-2-ib/code_review_agent.py
```

---

## 📂 Directory Structure

```text
projects/project-2-ib/
├── README.md                 # Project architecture & user guide
└── code_review_agent.py      # Core multi-pass review runner & Rich UI
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Framework:** LangChain & LangChain-Groq
- **Data Validation:** Pydantic v2
- **UI:** Rich (`Panel`, `Table`, `Syntax`, `Console`)
