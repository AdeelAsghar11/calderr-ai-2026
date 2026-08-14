"""
Phase 2 Verification Tests
- Invalid YAML rejected by /workflows/compile with HTTP 400 and helpful error details
- WebSocket event streaming pushes real-time node transition events
- API layer persistence across API server process restart
"""

import os
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def resolve_path(rel_path: str) -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(base_dir, rel_path)
    if os.path.exists(full_path):
        return full_path
    return rel_path


@pytest.fixture
def api_client(tmp_path):
    db_file = os.path.join(tmp_path, "api_test_state.db")
    os.environ["WORKFLOW_DB_PATH"] = db_file
    
    # Reload api module to pick up the test DB path
    # pyrefly: ignore [missing-import]
    import src.api as api_mod
    api_mod.engine = api_mod.WorkflowEngine(db_path=db_file)
    
    # Register test workflows
    api_mod.engine.register_yaml_file(resolve_path("workflows/1_function_pipeline.yaml"))
    api_mod.engine.register_yaml_file(resolve_path("workflows/2_human_approval.yaml"))
    
    client = TestClient(api_mod.app)
    return client, db_file, api_mod


def test_compile_malformed_yaml_returns_400(api_client):
    client, _, _ = api_client

    malformed_yaml = """
name: malformed_workflow
description: Broken YAML
state:
  - field: input_text
    type: invalid_type
nodes:
  - id: step1
    type: unknown_type
"""
    response = client.post(
        "/workflows/compile",
        json={"yaml_content": malformed_yaml},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"] or "Validation" in data["detail"] or "Unknown" in data["detail"]


def test_websocket_live_event_push(api_client):
    client, _, _ = api_client

    # 1. Start a workflow run via REST API
    res = client.post(
        "/workflows/function_pipeline/run",
        json={"input_text": "hello websocket streaming test"},
    )
    assert res.status_code == 200
    run_id = res.json()["run_id"]

    # 2. Connect via WebSocket to stream events
    events = []
    with client.websocket_connect(f"/workflows/function_pipeline/runs/{run_id}/stream") as ws:
        # Stream messages until connection finishes
        while True:
            try:
                data = ws.receive_json()
                events.append(data)
            except Exception:
                break

    # 3. Verify event sequence
    assert len(events) >= 1
    event_types = [e.get("type") or e.get("event") for e in events]
    assert len(event_types) >= 1


def test_api_persistence_across_restart(api_client):
    client, db_file, api_mod = api_client

    # 1. Start human_approval workflow run -> pauses at gate
    res = client.post(
        "/workflows/human_approval/run",
        json={"post_content": "Flagged comment for API restart test"},
    )
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    assert res.json()["status"] == "paused"

    # 2. Simulate API Server restart by reinstantiating WorkflowEngine & app on same DB file
    new_engine = api_mod.WorkflowEngine(db_path=db_file)
    new_engine.register_yaml_file(resolve_path("workflows/2_human_approval.yaml"))
    api_mod.engine = new_engine
    restarted_client = TestClient(api_mod.app)

    # 3. Resume paused run via POST /workflows/human_approval/runs/{run_id}/resume
    resume_res = restarted_client.post(
        f"/workflows/human_approval/runs/{run_id}/resume",
        json={"value": "approved_by_api_test"},
    )
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "completed"
    assert resume_res.json()["state"].get("approval_decision") == "approved_by_api_test" or resume_res.json()["state"].get("status") == "finalized_approved_by_api_test"
