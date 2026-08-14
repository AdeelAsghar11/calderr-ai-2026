"""
Phase 4 Pytest Verification Suite.

Tests final control plane, dashboard rendering, GitHub Agent dry-run, and graceful absence:
1. Control plane POST /runs and GET /runs/{run_id} via FastAPI TestClient.
2. Graceful-absence proof: Pipeline completes with status='completed', docker_build_result=None when Docker is unavailable.
3. Dry-run proof: GitHub Agent returns plan with actually_published=False and zero external API calls.
4. Dashboard render check: Streamlit dashboard view renders without errors when docker_build_result=None.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Ensure project directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from src.api import app
# pyrefly: ignore [missing-import]
from src.dashboard import render_dashboard_view
# pyrefly: ignore [missing-import]
from src.github_agent import run_github_agent
# pyrefly: ignore [missing-import]
from src.pipeline import run_full_pipeline
# pyrefly: ignore [missing-import]
from src.schema import GitHubPublishPlan, PipelineRun

FIXED_TODO_SPEC = (
    "A simple todo list API: users can add a todo, list all todos, "
    "mark a todo complete, and delete a todo."
)

client = TestClient(app)


def test_control_plane_endpoints(tmp_path):
    """
    Case 1: Full pipeline run via control plane.
    Triggers POST /runs and verifies GET /runs/{run_id} via TestClient.
    """
    post_res = client.post(
        "/runs",
        json={"feature_spec": FIXED_TODO_SPEC, "use_real": False},
    )
    assert post_res.status_code == 201, f"POST /runs failed: {post_res.text}"
    run_data = post_res.json()
    run_id = run_data["run_id"]
    assert run_data["status"] == "completed"
    assert "pm" in run_data["stages_completed"]
    assert "architect" in run_data["stages_completed"]

    get_res = client.get(f"/runs/{run_id}")
    assert get_res.status_code == 200, f"GET /runs/{run_id} failed: {get_res.text}"
    fetched_data = get_res.json()
    assert fetched_data["run_id"] == run_id
    assert fetched_data["status"] == "completed"


def test_graceful_absence_proof(tmp_path, monkeypatch):
    """
    Case 2: The Graceful-Absence Proof.
    Simulates Docker being unavailable (monkeypatching is_docker_available to False).
    Asserts pipeline completes cleanly with docker_build_result=None, validation_report=None,
    and 'devops'/'validation' absent from stages_completed.
    """
    monkeypatch.setattr("src.pipeline.is_docker_available", lambda: False)

    output_dir = str(tmp_path / "graceful_absence")
    pipeline_run = run_full_pipeline(
        feature_spec=FIXED_TODO_SPEC,
        output_dir=output_dir,
        use_real=False,
    )

    assert pipeline_run.status == "completed", "Pipeline failed when Docker was unavailable"
    assert pipeline_run.docker_build_result is None, "Expected docker_build_result to be None"
    assert pipeline_run.validation_report is None, "Expected validation_report to be None"
    assert "devops" not in pipeline_run.stages_completed
    assert "validation" not in pipeline_run.stages_completed
    assert "pm" in pipeline_run.stages_completed
    assert "security" in pipeline_run.stages_completed


def test_dry_run_proof(tmp_path, monkeypatch):
    """
    Case 3: The Dry-Run Proof.
    Runs GitHub Agent in dry-run mode (live=False).
    Asserts actually_published is False, zero network calls were made, and files_to_commit matches outputs.
    """
    monkeypatch.setattr("src.pipeline.is_docker_available", lambda: False)
    output_dir = str(tmp_path / "github_dry_run")
    pipeline_run = run_full_pipeline(FIXED_TODO_SPEC, output_dir, use_real=False)

    mock_client = MagicMock()
    plan: GitHubPublishPlan = run_github_agent(
        pipeline_run=pipeline_run,
        live=False,
        github_client=mock_client,
    )

    assert plan.dry_run is True
    assert plan.actually_published is False
    # Verify mock client was never invoked
    mock_client.create_repo_and_pr.assert_not_called()
    assert "backend.py" in plan.files_to_commit
    assert "index.html" in plan.files_to_commit


def test_dashboard_render_check(tmp_path, monkeypatch):
    """
    Case 4: Dashboard Render Check.
    Asserts Streamlit dashboard view function renders cleanly without raising exceptions when
    passed a PipelineRun object with docker_build_result=None.
    """
    monkeypatch.setattr("src.pipeline.is_docker_available", lambda: False)
    output_dir = str(tmp_path / "dashboard_check")
    pipeline_run = run_full_pipeline(FIXED_TODO_SPEC, output_dir, use_real=False)

    # Should execute render_dashboard_view without raising an exception
    try:
        render_dashboard_view(pipeline_run)
    except Exception as e:
        pytest.fail(f"Dashboard render failed on PipelineRun with docker_build_result=None: {e}")
