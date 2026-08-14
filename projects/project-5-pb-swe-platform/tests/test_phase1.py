"""
Phase 1 Pytest Verification Suite.

Tests offline stub pipeline execution for the fixed Todo List feature spec:
1. TaskList contains >= 3 tasks (each with >= 1 criterion) and SystemContract contains >= 3 endpoints.
2. Backend file is written to disk and passes Python AST syntax parsing.
3. Contract consistency proof: every EndpointSpec.path appears in both backend and frontend files.
4. IntegrationResult.consistent is True with zero missing endpoints.
"""

import ast
import os
import sys
# pyrefly: ignore [missing-import]
import pytest

# Insert project directory to sys.path so src imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from src.pipeline import run_phase1_pipeline
# pyrefly: ignore [missing-import]
from src.schema import SystemContract, TaskList

FIXED_TODO_SPEC = (
    "A simple todo list API: users can add a todo, list all todos, "
    "mark a todo complete, and delete a todo."
)


@pytest.fixture
def pipeline_run_result(tmp_path):
    """
    Fixture executing the Phase 1 pipeline against the fixed todo spec using offline stub mode.
    Outputs files into pytest's isolated tmp_path.
    """
    output_dir = str(tmp_path / "generated_code")
    result = run_phase1_pipeline(
        feature_spec=FIXED_TODO_SPEC,
        output_dir=output_dir,
        use_real=False,
    )
    return result


def test_pm_and_architect_outputs(pipeline_run_result):
    """
    Requirement 1: Verify PM Agent produces >= 3 tasks with criteria and Architect Agent produces >= 3 endpoints.
    """
    task_list: TaskList = pipeline_run_result["task_list"]
    contract: SystemContract = pipeline_run_result["contract"]

    assert len(task_list.tasks) >= 3, f"Expected at least 3 tasks, got {len(task_list.tasks)}"
    for task in task_list.tasks:
        assert len(task.acceptance_criteria) >= 1, f"Task '{task.description}' missing acceptance criteria"

    assert len(contract.endpoints) >= 3, f"Expected at least 3 endpoints, got {len(contract.endpoints)}"


def test_backend_syntax_validity(pipeline_run_result):
    """
    Requirement 2: Verify generated backend file is written to disk and is syntactically valid Python.
    """
    backend_comp = pipeline_run_result["backend"]
    assert os.path.exists(backend_comp.file_path), "Backend file was not written to disk"

    with open(backend_comp.file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse file contents with AST to ensure syntactically valid Python code
    parsed_ast = ast.parse(content)
    assert parsed_ast is not None, "Failed to parse generated backend code into AST"


def test_contract_consistency_proof(pipeline_run_result):
    """
    Requirement 3: Contract consistency proof.
    For every EndpointSpec.path in SystemContract, assert path string appears in backend file AND frontend file.
    """
    contract: SystemContract = pipeline_run_result["contract"]
    backend_comp = pipeline_run_result["backend"]
    frontend_comp = pipeline_run_result["frontend"]

    with open(backend_comp.file_path, "r", encoding="utf-8") as f:
        backend_content = f.read()

    with open(frontend_comp.file_path, "r", encoding="utf-8") as f:
        frontend_content = f.read()

    for endpoint in contract.endpoints:
        path = endpoint.path
        assert path in backend_content, (
            f"Contract endpoint path '{path}' missing from generated backend file ({backend_comp.file_path})"
        )
        assert path in frontend_content, (
            f"Contract endpoint path '{path}' missing from generated frontend file ({frontend_comp.file_path})"
        )


def test_integration_result_consistent(pipeline_run_result):
    """
    Requirement 4: Verify IntegrationResult.consistent is True with empty missing endpoint lists.
    """
    integration_result = pipeline_run_result["integration"]
    assert integration_result.consistent is True, f"Integration failed: {integration_result.notes}"
    assert len(integration_result.missing_in_backend) == 0, f"Backend missing: {integration_result.missing_in_backend}"
    assert len(integration_result.missing_in_frontend) == 0, f"Frontend missing: {integration_result.missing_in_frontend}"
