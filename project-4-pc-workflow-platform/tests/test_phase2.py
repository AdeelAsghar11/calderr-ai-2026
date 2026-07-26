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


@pytest.fixture
def api_client(tmp_path):
    db_file = os.path.join(tmp_path, "api_test_state.db")
    os.environ["WORKFLOW_DB_PATH"] = db_file
    
    # Reload api module to pick up the test DB path
    import src.api as api_mod
    api_mod.engine = api_mod.WorkflowEngine(db_path=db_file)
    
    # Register test workflows
    api_mod.engine.register_yaml_file("project-4-pc-workflow-platform/workflows/1_function_pipeline.yaml")
    api_mod.engine.register_yaml_file("project-4-pc-workflow-platform/workflows/2_human_approval.yaml")
    
    client = TestClient(api_mod.app)
    return client, db_file, api_mod


def test_compile_malformed_yaml_returns_400(api_client):
    client, _, _ = api_client

    malformed_yaml = """
name: malformed_workflow
description: Broken YAML
state:
  - field: input_text
    type: str
nodes:
  - id: node_1
    type: function
    # missing function_name!
edges:
  - from: START
    to: node_1
"""
    response = client.post(
        "/workflows/compile",
        json={"yaml_content": malformed_yaml},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "YAML Schema Validation Error" in detail
    assert "must specify 'function_name'" in detail or "node_1" in detail


def test_websocket_live_event_push(api_client):
    client, _, _ = api_client

    # 1. Start human_approval run via POST /workflows/human_approval/run
    res = client.post(
        "/workflows/human_approval/run",
        json={"initial_state": {"post_content": "Check stream content"}},
    )
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    assert res.json()["status"] == "paused"

    # 2. Connect WebSocket to stream endpoint
    with client.websocket_connect(f"/workflows/human_approval/runs/{run_id}/stream") as websocket:
        msg = websocket.receive_json()
        assert msg["event"] == "paused"
        assert msg["current_node"] == "human_review_gate"
        assert msg["state"]["post_content"] == "Check stream content"


def test_api_persistence_across_restart(api_client):
    client, db_file, api_mod = api_client

    # 1. Start human approval run on API Server 1
    res = client.post(
        "/workflows/human_approval/run",
        json={"initial_state": {"post_content": "Server restart test"}},
    )
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    assert res.json()["status"] == "paused"

    # 2. Simulate API Server restart by reinstantiating WorkflowEngine & app on same DB file
    new_engine = api_mod.WorkflowEngine(db_path=db_file)
    new_engine.register_yaml_file("project-4-pc-workflow-platform/workflows/2_human_approval.yaml")
    api_mod.engine = new_engine
    restarted_client = TestClient(api_mod.app)

    # 3. Resume paused run via POST /workflows/human_approval/runs/{run_id}/resume
    resume_res = restarted_client.post(
        f"/workflows/human_approval/runs/{run_id}/resume",
        json={"value": "approved_by_api_test"},
    )
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "completed"
    assert resume_res.json()["state"]["review_status"] == "finalized_approved_by_api_test"

    # 4. Verify GET endpoint returns completed status
    get_res = restarted_client.get(f"/workflows/human_approval/runs/{run_id}")
    assert get_res.status_code == 200
    assert get_res.json()["status"] == "completed"
