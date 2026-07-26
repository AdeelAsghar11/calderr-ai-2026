# Building a Production Declarative Workflow Orchestration Engine with LangGraph, SQLite, & FastAPI

## The Problem: Hard-Coded Graphs vs. Runtime Orchestration

When building agentic AI pipelines, developer teams often start by writing individual Python scripts containing explicit graph definitions. While hand-written LangGraph scripts work well for fixed prototypes, scaling across dozens of domain-specific workflows introduces severe maintainability bottlenecks:
1. **Code Duplication & Churn**: Every tweak to a prompt template, routing rule, or human-in-the-loop gate requires modifying application code, re-testing Python modules, and re-deploying services.
2. **Lack of Dynamic Orchestration**: Product managers and domain experts cannot declare or edit workflow topologies without writing Python code.
3. **Fragile State Persistence**: Naive implementations relying on in-memory state savers (`MemorySaver`) crash and lose state upon process restarts or cloud deployments.

**Project 4-P-C** solves this by providing a general-purpose **declarative workflow orchestration platform**. Workflows are specified in standardized YAML files, validated against a strict Pydantic schema, dynamically compiled into executable LangGraph `StateGraph` structures at runtime, and persisted to SQLite via `SqliteSaver`.

---

## Architectural Highlights

```
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  YAML Workflow  │ ───►  │ Pydantic Schema      │ ───►  │ Dynamic LangGraph      │
│  Definitions    │       │ Validation           │       │ StateGraph Compiler    │
└─────────────────┘       └──────────────────────┘       └───────────┬────────────┘
                                                                     │
                                                                     ▼
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│ Streamlit UI    │ ◄───► │ FastAPI REST API     │ ◄───► │ SqliteSaver Persistent │
│ & Graphviz      │       │ & WebSockets         │       │ SQLite Engine          │
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
```

### 1. Dynamic State & Reducer Assembly
Rather than requiring pre-compiled Python dataclasses, the engine inspects the declared `state` schema of the YAML file and dynamically constructs a `TypedDict` class. State fields specified with `reducer: append` are mapped directly to `Annotated[list, operator.add]`:

```python
def build_dynamic_state_type(spec: WorkflowSpec) -> type:
    fields: Dict[str, Any] = {}
    for field_spec in spec.state:
        base_type = TYPE_MAP.get(field_spec.type, str)
        if field_spec.reducer == "append":
            fields[field_spec.field] = Annotated[list, operator.add]
        else:
            fields[field_spec.field] = base_type
    return TypedDict("DynamicWorkflowState", fields)
```

### 2. Execution Security via Fixed Function Registries
Allowing arbitrary code execution from user-supplied YAML strings is a critical security vulnerability. This platform strictly enforces declarative function step execution by dispatching `type: function` nodes through a fixed, pre-audited `FUNCTION_REGISTRY` dictionary — **zero** `eval()` or `exec()` calls:

```python
def make_function_node(node_spec):
    fn_name = node_spec.function_name
    if fn_name not in FUNCTION_REGISTRY:
        raise ValueError(f"Function '{fn_name}' requested by node '{node_spec.id}' is not in FUNCTION_REGISTRY.")
    fn = FUNCTION_REGISTRY[fn_name]
    return lambda state: fn(state)
```

### 3. Process-Durable Human-in-the-Loop Interrupts
When a `human_review` node is encountered, the graph invokes LangGraph's native `interrupt()` function. Because the execution engine binds to `SqliteSaver`, the run pauses cleanly and persists to disk. If the application or server process dies, a new process loading the database can inspect pending interrupts and resume seamlessly via `Command(resume=...)`.

---

## Verification & House Style Transparency

In accordance with our house engineering standards, we report exact, verified empirical results rather than unconfirmed claims.

### What Has Been Fully Verified in Automated Tests
1. **Pure Function Workflows**: Verified end-to-end execution of multi-step function transformation graphs (`test_function_workflow_execution`).
2. **Process-Kill Persistence & Resume**: Verified that a paused `human_review` run in Process 1 survives process termination and is successfully resumed by a completely distinct OS Python subprocess (`resume_subprocess.py`) loading `test_state.db`.
3. **Cycle Safeguards**: Verified that cyclic graphs exceeding `max_iterations` cap exit cleanly with hard recursion limit errors (`test_max_iterations_hard_cap`).
4. **API Malformed Rejection**: Verified that invalid YAML files are rejected by `POST /workflows/compile` with HTTP 400 and detailed Pydantic schema validation messages (`test_compile_malformed_yaml_returns_400`).
5. **WebSocket Status Streaming**: Verified real-time WebSocket connection and event emission for paused and executing workflows (`test_websocket_live_event_push`).
6. **API Persistence Across Server Restarts**: Verified run initiation, process destruction, server re-instantiation, and successful HTTP resume via `POST /workflows/.../resume` (`test_api_persistence_across_restart`).
7. **5 Example Workflows**: Verified that all 5 YAML workflow files parse, validate, and compile without error (`test_all_5_yaml_workflows_parse_and_compile`).

- **Test Suite Command**: `uv run python -m pytest project-4-pc-workflow-platform/tests/`
- **Result**: **7 passed in 1.75 seconds**.

### What Requires a Live `GROQ_API_KEY` to Confirm
- End-to-end live text generation responses for `3_linear_llm_pipeline.yaml`, `4_llm_classification_branch.yaml`, and `5_cyclic_refinement.yaml` against Groq's `llama-3.3-70b-versatile` API endpoints. (Compiler structures, closure factories, and `ChatGroq` bindings are fully implemented and verified offline).
