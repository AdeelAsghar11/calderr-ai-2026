# Project 2-P-C: Automated Data Analysis Agent

An interactive data analysis platform built with **Streamlit**, **Pandas**, **Matplotlib**, **LangChain**, and **Groq (`llama-3.3-70b-versatile`)**.

The agent accepts natural language questions over uploaded CSV files or pre-loaded datasets (Sales, Employees, Student Grades), dynamically writes executable Pandas/Matplotlib Python code, runs execution in a controlled environment, and displays data summaries, computational answers, and plots.

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │  User Question & CSV   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Code Generation Agent │
                       │  (Pandas / Matplotlib) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Execution Harness      │
                       │ (Controlled exec context)│
                       └───────────┬────────────┘
                         │                    │
           ┌─────────────┴──────┐      ┌──────┴─────────────┐
           ▼                    ▼      ▼                    ▼
     [Success output]        [Plot] [Errors]        [Self-Correction]
           │                    │      │                    │
           └─────────────┬──────┘      └──────┬─────────────┘
                         ▼                    ▼
               ┌──────────────────────────────────┐
               │  Streamlit Interactive Dashboard  │
               └──────────────────────────────────┘
```

---

## ✨ Features

- **Natural Language to Pandas:** Converts plain-English queries into data manipulation logic (`df.groupby()`, `df.describe()`, `df.corr()`).
- **Dynamic Data Visualization:** Generates and displays Matplotlib line charts, bar plots, histograms, and scatter plots based on user queries.
- **Built-in Sample Datasets:** Includes pre-loaded datasets (Sales Data, Employee Data, Student Grades) for instant testing.
- **Error Recovery Loop:** Captures execution tracebacks and re-prompts the LLM to fix syntax or runtime errors.
- **Interactive Streamlit UI:** Data frame previews, generated code drawer, text responses, and rendered plots.

---

## 🚀 Quick Start & Usage

### Prerequisites
- Python 3.11+
- Groq API Key set in `.env` (`GROQ_API_KEY`)

### Launch Streamlit Dashboard
From the repository root:
```bash
uv run streamlit run projects/project-2-pc/data_analysis_agent.py
```

---

## 📂 Directory Structure

```text
projects/project-2-pc/
├── README.md               # Project overview & documentation
├── data_analysis_agent.py  # Streamlit dashboard & execution loop
└── requirements_2pc.txt    # Project dependencies
```

---

## 💻 Tech Stack

- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend / Dashboard:** Streamlit
- **Data & Plotting:** Pandas, NumPy, Matplotlib
- **Framework:** LangChain & LangChain-Groq
