"""
FastAPI REST API and WebSocket streaming server for Project 4-P-C.
"""

from __future__ import annotations

import os
import sys
import uuid

# Ensure project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, Optional
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from src.compiler import compile_workflow, compile_workflow_from_yaml
# pyrefly: ignore [missing-import]
from src.engine import WorkflowEngine
# pyrefly: ignore [missing-import]
from src.schema import WorkflowSpec

from contextlib import asynccontextmanager

DB_PATH = os.environ.get("WORKFLOW_DB_PATH", "workflows_api_state.db")
engine = WorkflowEngine(db_path=DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    workflows_dir = os.path.join(os.path.dirname(__file__), "..", "workflows")
    if os.path.exists(workflows_dir):
        for fname in os.listdir(workflows_dir):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(workflows_dir, fname)
                try:
                    engine.register_yaml_file(fpath)
                except Exception as e:
                    print(f"Warning: Failed to pre-register {fname}: {e}")
    yield


app = FastAPI(
    title="Workflow Orchestration Platform API",
    description="YAML-compiled LangGraph engine with SQLite persistence & WebSocket status streaming",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompileRequest(BaseModel):
    yaml_content: str = Field(..., description="Raw YAML workflow definition")


class RunRequest(BaseModel):
    initial_state: Dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    value: Any = Field(..., description="Resume payload value for human_review interrupt")


@app.post("/workflows/compile", status_code=201)
async def compile_workflow_endpoint(request: Request):
    """Compiles raw YAML into a validated workflow graph."""
    content_type = request.headers.get("content-type", "")
    yaml_text = ""
    
    if "application/json" in content_type:
        body = await request.json()
        yaml_text = body.get("yaml_content", "") or body.get("yaml", "")
    else:
        body_bytes = await request.body()
        yaml_text = body_bytes.decode("utf-8")

    if not yaml_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Validation Error: YAML content body cannot be empty.",
        )

    try:
        spec = engine.register_yaml(yaml_text)
        return {
            "workflow_id": spec.name,
            "node_count": len(spec.nodes),
            "description": spec.description,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"YAML Schema Validation Error: {str(e)}",
        )


@app.get("/workflows")
def list_workflows():
    """Lists all registered workflows."""
    specs = engine.list_workflows()
    return [
        {
            "workflow_id": name,
            "description": spec.description,
            "node_count": len(spec.nodes),
            "max_iterations": spec.max_iterations,
        }
        for name, spec in specs.items()
    ]


@app.post("/workflows/{workflow_id}/run")
def run_workflow(workflow_id: str, request: RunRequest):
    """Starts execution of a workflow run."""
    spec = engine.get_workflow_spec(workflow_id)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    thread_id = str(uuid.uuid4())
    try:
        result = engine.run_workflow(workflow_id, request.initial_state, thread_id)
        return {
            "run_id": thread_id,
            "status": result["status"],
            "current_node": result.get("current_node"),
            "state": result.get("state"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


@app.get("/workflows/{workflow_id}/runs/{run_id}")
def get_run_status(workflow_id: str, run_id: str):
    """Gets the current status and state of a run."""
    spec = engine.get_workflow_spec(workflow_id)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    status_data = engine.get_run_status(workflow_id, run_id)
    if status_data["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    return status_data


@app.post("/workflows/{workflow_id}/runs/{run_id}/resume")
def resume_workflow(workflow_id: str, run_id: str, request: ResumeRequest):
    """Resumes a paused human_review node with user response."""
    spec = engine.get_workflow_spec(workflow_id)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    try:
        result = engine.resume_workflow(workflow_id, run_id, request.value)
        return {
            "run_id": run_id,
            "status": result["status"],
            "current_node": result.get("current_node"),
            "state": result.get("state"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resume error: {str(e)}")


@app.websocket("/workflows/{workflow_id}/runs/{run_id}/stream")
async def stream_workflow_run(websocket: WebSocket, workflow_id: str, run_id: str):
    """Pushes real-time WebSocket events on node transitions and run state changes."""
    await websocket.accept()
    spec = engine.get_workflow_spec(workflow_id)
    if not spec:
        await websocket.send_json({"event": "error", "message": f"Workflow '{workflow_id}' not found"})
        await websocket.close()
        return

    # Check current status first
    current_status = engine.get_run_status(workflow_id, run_id)
    if current_status["status"] in ("paused", "completed"):
        await websocket.send_json({
            "event": current_status["status"],
            "current_node": current_status.get("current_node"),
            "state": current_status["state"],
        })
        await websocket.close()
        return

    # Emit initial state event
    await websocket.send_json({"event": "connected", "run_id": run_id, "status": current_status["status"]})

    # Stream updates using graph.astream
    # pyrefly: ignore [missing-import]
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(engine.db_path) as checkpointer:
        graph = compile_workflow(spec, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": run_id}}

        try:
            # Iterate through execution stream
            async for chunk in graph.astream(None, config=config, stream_mode="updates"):
                for node_name, state_update in chunk.items():
                    await websocket.send_json({
                        "event": "node_transition",
                        "node": node_name,
                        "update": state_update,
                    })

            # Check final status after stream ends
            updated_status = engine.get_run_status(workflow_id, run_id)
            if updated_status["status"] == "paused":
                await websocket.send_json({
                    "event": "paused",
                    "current_node": updated_status["current_node"],
                    "state": updated_status["state"],
                })
            elif updated_status["status"] == "completed":
                await websocket.send_json({
                    "event": "completed",
                    "state": updated_status["state"],
                })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.send_json({"event": "error", "message": str(e)})
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
