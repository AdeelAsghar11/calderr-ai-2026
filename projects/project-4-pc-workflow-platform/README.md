# Project 4-P-C: Production Workflow Orchestration Platform

A general-purpose production orchestration platform that compiles declarative YAML workflow definitions into executable LangGraph state graphs at runtime with SQLite-backed process-durable persistence (`SqliteSaver`), REST API, WebSocket status streaming, and an interactive Streamlit dashboard.

---

## Key Features

- **Runtime Dynamic YAML Compilation**: Compiles declared state fields, nodes (`llm_call`, `function`, `human_review`), edges, and conditional routing into a compiled LangGraph `StateGraph` dynamically at runtime without hand-writing Python graph files for each workflow.
- **Process-Kill Persistence**: Uses `SqliteSaver` against disk databases to ensure pending Human-in-the-Loop (`human_review`) approvals survive process kills, app crashes, and server restarts.
- **Strict Execution Safety**: Function nodes reference a fixed, safe `FUNCTION_REGISTRY`. **Zero** `eval` or `exec` on YAML code strings.
- **FastAPI REST API & WebSocket Streaming**: Full REST contract (`/workflows/compile`, `/workflows`, `/workflows/{id}/run`, `/workflows/{id}/runs/{run_id}`, `/workflows/{id}/runs/{run_id}/resume`) with real-time WebSocket event streaming (`WS /workflows/{id}/runs/{run_id}/stream`).
- **Streamlit Dashboard**: Graphviz visualizer rendering workflow state diagrams, live run monitoring, human approval form, and dynamic YAML live compilation.
- **Cycle Control & Safeguards**: `max_iterations` enforced as a hard recursion cap on cyclic graphs.

---

## Directory Structure

```text
projects/project-4-pc-workflow-platform/
├── README.md
├── BLOG.md
├── Dockerfile
├── docker-compose.yml
├── workflows/
│   ├── 1_function_pipeline.yaml           # Pure function transformation
│   ├── 2_human_approval.yaml              # Human-in-the-loop content review
│   ├── 3_linear_llm_pipeline.yaml         # Multi-step LLM text generator & summarizer
│   ├── 4_llm_classification_branch.yaml   # LLM classifier -> conditional routing
│   └── 5_cyclic_refinement.yaml          # Cyclic function check -> LLM rewrite + max_iterations
├── src/
│   ├── schema.py                          # Pydantic schema validation for YAML workflows
│   ├── registry.py                        # Fixed FUNCTION_REGISTRY dict
│   ├── compiler.py                        # YAML -> LangGraph StateGraph dynamic compiler
│   ├── engine.py                          # SqliteSaver execution engine
│   ├── api.py                             # FastAPI REST API + WebSocket streaming server
│   └── dashboard.py                       # Streamlit UI dashboard & Graphviz renderer
└── tests/
    ├── test_phase1.py                     # Compiler, SqliteSaver & sub-process restart persistence
    ├── test_phase2.py                     # REST API, 400 error rejection, WS push & API restart resume
    └── test_phase3.py                     # Validation for all 5 YAML workflow examples
```

---

## Quick Start & Setup

### Prerequisites
- Python >= 3.11
- Dependency management with `uv`
- `GROQ_API_KEY` in environment or `.env` for LLM nodes (`llama-3.3-70b-versatile`)

### 1. Run Automated Test Suite
```bash
uv run python -m pytest projects/project-4-pc-workflow-platform/tests/
```

### 2. Launch FastAPI Server
```bash
uv run uvicorn projects.project-4-pc-workflow-platform.src.api:app --reload --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`

### 3. Launch Streamlit Dashboard
```bash
uv run streamlit run projects/project-4-pc-workflow-platform/src/dashboard.py
```
- Access Dashboard: `http://localhost:8501`

### 4. Run via Docker Compose
```bash
docker-compose -f projects/project-4-pc-workflow-platform/docker-compose.yml up --build
```

---

## YAML Workflow Schema

Workflows are declared using the following schema format:

```yaml
name: my_workflow
description: Workflow description

state:
  - field: input_text
    type: str
    reducer: overwrite          # overwrite (default) or append (Annotated[list, operator.add])
    default: "initial value"
  - field: logs
    type: list
    reducer: append

nodes:
  - id: step_1
    type: function
    function_name: count_words   # MUST match an entry in FUNCTION_REGISTRY

  - id: step_2
    type: llm_call
    prompt_template: "Summarize: {input_text}"
    output_field: summary_text
    temperature: 0.2

  - id: approval_gate
    type: human_review
    payload_fields: [input_text, summary_text]
    resume_field: approval_decision

edges:
  - from: START
    to: step_1
  - from: step_1
    to: step_2
  - from: step_2
    to: approval_gate
  - from: approval_gate
    to: END

max_iterations: 10              # Optional hard limit cap on cycles
```

---

## REST & WebSocket API Contract

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/workflows/compile` | Validates raw YAML body, returns `{workflow_id, node_count}` or HTTP 400 with validation error |
| `GET` | `/workflows` | Lists registered workflows |
| `POST` | `/workflows/{id}/run` | Starts run with initial state dict, returns `{run_id, status}` |
| `GET` | `/workflows/{id}/runs/{run_id}` | Gets current status (`running` \| `paused` \| `completed` \| `failed`), current node, and state |
| `POST` | `/workflows/{id}/runs/{run_id}/resume` | Resumes paused `human_review` run with `{value: ...}` |
| `WS` | `/workflows/{id}/runs/{run_id}/stream` | Real-time WebSocket push of node transitions and status events |
