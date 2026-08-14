"""
FastAPI REST Control Plane and WebSocket progress streaming server.

Provides HTTP management endpoints for triggering pipeline executions, querying run history,
and streaming stage-by-stage execution updates.
"""

import os
import sys
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_full_pipeline
from src.schema import PipelineRun

app = FastAPI(
    title="AI Software Platform Control Plane API",
    description="REST control plane and WebSocket streaming server for end-to-end AI software engineering pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory pipeline runs storage database
RUNS_DB: Dict[str, PipelineRun] = {}


class CreateRunRequest(BaseModel):
    feature_spec: str = Field(..., description="Natural language feature specification input")
    use_real: bool = Field(default=False, description="Toggle live ChatGroq LLM execution vs stub mode")


@app.post("/runs", response_model=PipelineRun, status_code=201)
def create_run(request: CreateRunRequest):
    """
    Triggers an end-to-end multi-agent pipeline run for the given feature spec.
    """
    if not request.feature_spec.strip():
        raise HTTPException(status_code=400, detail="Feature spec cannot be empty.")

    run_id = str(uuid.uuid4())
    output_dir = os.path.abspath(os.path.join("generated_runs", run_id))

    try:
        pipeline_run = run_full_pipeline(
            feature_spec=request.feature_spec,
            output_dir=output_dir,
            use_real=request.use_real,
            run_id=run_id,
        )
        RUNS_DB[run_id] = pipeline_run
        return pipeline_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.get("/runs", response_model=List[PipelineRun])
def list_runs():
    """
    Returns a list of all historical pipeline runs.
    """
    return list(RUNS_DB.values())


@app.get("/runs/{run_id}", response_model=PipelineRun)
def get_run(run_id: str):
    """
    Retrieves detailed execution state for a specific run_id.
    """
    if run_id not in RUNS_DB:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found.")
    return RUNS_DB[run_id]


@app.websocket("/ws/runs/{run_id}")
async def stream_run_progress(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint streaming stage progress updates as pipeline stages execute.
    """
    await websocket.accept()
    try:
        if run_id in RUNS_DB:
            run = RUNS_DB[run_id]
            for stage in run.stages_completed:
                await websocket.send_json({"stage": stage, "status": "completed"})
            await websocket.send_json({"status": run.status, "completed": True})
        else:
            await websocket.send_json({"error": "Run ID not found", "status": "failed"})
    except WebSocketDisconnect:
        pass
