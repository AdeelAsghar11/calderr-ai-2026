# Project 5-P-B: End-to-End AI Software Engineering Platform

A multi-agent AI software engineering platform built with **Python 3.11+**, **FastAPI**, **Streamlit**, **Docker**, **Pydantic**, **Pytest**, **LangChain**, and **Groq (`llama-3.3-70b-versatile`)**.

The system converts natural language feature specifications into fully generated, contract-validated, tested, OWASP-audited, containerized, and published web applications.

---

## Key Features

- **Contract-First Code Generation**: PM Agent decomposes feature specs into structured tasks with testable acceptance criteria, and Architect Agent establishes a typed `SystemContract` that constrains both Backend and Frontend generation.
- **Contract-Consistency Verification**: Integration Agent performs non-generative static path matching to prove generated FastAPI backend code and HTML/JS frontend code reference identical API contracts.
- **Sandboxed Subprocess QA Execution**: QA Agent generates real pytest test functions per acceptance criterion and executes them inside isolated sandboxed subprocesses (`tempfile.TemporaryDirectory()`, wall-clock timeouts, memory/CPU caps, network isolation).
- **OWASP Static Security Auditing**: Security Agent scans generated components for SQL injection, hardcoded credentials, missing parameter validation, and DOM XSS vulnerabilities, classifying findings into structured severity ratings.
- **Automated DevOps & Runtime Validation**: DevOps Agent builds Docker containers (`python:3.11-slim`), and Validation Agent dynamically allocates free host ports (`socket.bind(('127.0.0.1', 0))`) to test stateful HTTP workflows (POST -> GET -> PUT -> DELETE -> GET) with guaranteed `finally:` container teardown.
- **Dry-Run Gated GitHub Publishing**: GitHub Agent prepares repository trees and PR metadata in dry-run mode by default, requiring explicit `--live` activation and `GITHUB_TOKEN` before making external API calls.
- **FastAPI Control Plane & Streamlit UI**: REST control plane (`POST /runs`, `GET /runs`, WebSocket status streaming) paired with an interactive Streamlit dashboard.

---

## Directory Structure

```text
projects/project-5-pb-swe-platform/
├── BLOG.md
├── README.md
├── src/
│   ├── __init__.py
│   ├── schema.py              # Pydantic v2 domain & verification models
│   ├── pm_agent.py            # PM Agent (task decomposition & acceptance criteria)
│   ├── architect_agent.py     # Architect Agent (typed SystemContract design)
│   ├── backend_agent.py       # Backend Agent (FastAPI code generation to disk)
│   ├── frontend_agent.py      # Frontend Agent (HTML/JS code generation to disk)
│   ├── integration_agent.py   # Integration Agent (non-generative contract matching)
│   ├── qa_agent.py           # QA Agent (sandboxed pytest writing & execution)
│   ├── security_agent.py      # Security Agent (static vulnerability scanner)
│   ├── devops_agent.py        # DevOps Agent (Dockerfile & docker-compose builds)
│   ├── validation_agent.py    # Validation Agent (live container HTTP testing & cleanup)
│   ├── github_agent.py        # GitHub Agent (dry-run repository & PR preparation)
│   ├── pipeline.py            # End-to-end multi-agent pipeline orchestrator
│   ├── api.py                 # FastAPI control plane server & WebSocket endpoint
│   └── dashboard.py           # Streamlit interactive dashboard UI
└── tests/
    ├── test_phase1.py         # Phase 1 codegen & contract consistency tests
    ├── test_phase2.py         # Phase 2 QA sandboxing & security scanner tests
    ├── test_phase3.py         # Phase 3 Docker build & container validation tests
    └── test_phase4.py         # Phase 4 Control plane, dashboard & dry-run tests
```

---

## Multi-Agent Pipeline Architecture

```text
 feature spec (text)
       |
       v
   PM Agent  -----> TaskList (tasks + acceptance criteria)
       |
       v
Architect Agent ---> SystemContract (typed endpoint specs & models)
       |
  +----+----+
  |         |
  v         v
Backend   Frontend
 Agent     Agent
 (py)     (html)
  |         |
  +----+----+
       |
       v
Integration Agent --> Contract Consistency Proof
       |
       v
   QA Agent --------> Sandboxed Pytest Execution Report
       |
       v
Security Agent -----> OWASP Vulnerability Severity Report
       |
       v
 DevOps Agent ------> Dockerfile & docker build Execution
       |
       v
Validation Agent ---> Live Container HTTP Verification & Teardown
       |
       v
 GitHub Agent ------> Dry-Run Repository & Pull Request Plan
```

---

## Quick Start & Setup

### 1. Run Automated Test Suite (Phases 1 - 4)

```bash
uv run python -m pytest projects/project-5-pb-swe-platform/tests/ -v
```

### 2. Launch FastAPI Control Plane

```bash
uv run uvicorn projects.project-5-pb-swe-platform.src.api:app --host 127.0.0.1 --port 8000
```

Trigger a pipeline run via REST:
```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d "{\"feature_spec\": \"A simple todo list API: users can add a todo, list all todos, mark a todo complete, and delete a todo.\"}"
```

### 3. Launch Streamlit Interactive Dashboard

```bash
uv run streamlit run projects/project-5-pb-swe-platform/src/dashboard.py
```
