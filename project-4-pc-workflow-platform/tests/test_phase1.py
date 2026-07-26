"""
Phase 1 Verification Tests
- Pure function workflow end-to-end execution
- Human-in-the-loop pause & resume across a REAL OS PROCESS RESTART
- Max iterations hard cap enforcement on cyclic graph
"""

import os
import subprocess
import sys
import uuid
import pytest

# Add project directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import WorkflowEngine


@pytest.fixture
def engine(tmp_path):
    db_file = os.path.join(tmp_path, "test_state.db")
    eng = WorkflowEngine(db_path=db_file)
    eng.register_yaml_file("project-4-pc-workflow-platform/workflows/1_function_pipeline.yaml")
    eng.register_yaml_file("project-4-pc-workflow-platform/workflows/2_human_approval.yaml")
    eng.register_yaml_file("project-4-pc-workflow-platform/workflows/test_cyclic.yaml")
    return eng, db_file


def test_function_workflow_execution(engine):
    eng, _ = engine
    thread_id = str(uuid.uuid4())
    result = eng.run_workflow(
        "function_pipeline",
        initial_state={"input_text": "hello world from pytest"},
        thread_id=thread_id,
    )

    assert result["status"] == "completed"
    assert result["state"]["word_count"] == 4
    assert result["state"]["transformed_text"] == "HELLO WORLD FROM PYTEST"
    assert len(result["state"]["logs"]) == 2


def test_human_review_process_restart_persistence(engine):
    eng, db_file = engine
    thread_id = str(uuid.uuid4())

    # Process 1 (current process): Run workflow until interrupt
    result = eng.run_workflow(
        "human_approval",
        initial_state={"post_content": "Flagged user content needing review"},
        thread_id=thread_id,
    )

    assert result["status"] == "paused"
    assert result["current_node"] == "human_review_gate"

    # Process 2: Execute a completely separate OS python process to resume from SQLite DB file
    cmd = [
        sys.executable,
        "project-4-pc-workflow-platform/tests/resume_subprocess.py",
        db_file,
        thread_id,
        "approved",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    assert proc.returncode == 0, f"Subprocess failed with stderr: {proc.stderr}"
    assert "RESUME_SUCCESS" in proc.stdout
    assert "finalized_approved" in proc.stdout

    # Verify final state in the current process by reading from disk checkpointer
    post_resume_status = eng.get_run_status("human_approval", thread_id)
    assert post_resume_status["status"] == "completed"
    assert post_resume_status["state"]["review_status"] == "finalized_approved"
    assert post_resume_status["state"]["approval_decision"] == "approved"


def test_max_iterations_hard_cap(engine):
    eng, _ = engine
    thread_id = str(uuid.uuid4())

    result = eng.run_workflow(
        "cyclic_test",
        initial_state={"iteration_count": 0},
        thread_id=thread_id,
    )

    assert result["status"] == "failed"
    assert "max_iterations limit" in result["error"]
