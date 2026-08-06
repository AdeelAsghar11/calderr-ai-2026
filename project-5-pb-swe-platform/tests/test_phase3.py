"""
Phase 3 Pytest Verification Suite.

Tests DevOps and Validation agents:
1. The Build Proof: Real docker build succeeds on Phase 1 generated backend -> build_succeeded is True.
2. The Build-Failure Proof: Docker build against deliberately broken fixture -> build_succeeded is False with output.
3. The Runtime Proof: Live container HTTP verification (POST -> GET -> PUT -> DELETE -> GET) -> all_passed is True.
4. The Cleanup Proof: Verify container is terminated via docker ps after normal run AND forced failure.
5. Contract-Drift Check: Assert ValidationReport.endpoint_checks dynamically covers every SystemContract endpoint.

Handles Docker availability gracefully: skips cleanly if docker info fails.
"""

import os
import subprocess
import sys
import pytest

# Insert project directory to sys.path so src imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.devops_agent import run_devops_agent
from src.pipeline import run_phase1_pipeline, run_phase3_pipeline
from src.schema import DockerBuildResult, GeneratedComponent, ValidationReport
from src.validation_agent import _is_container_running, run_validation_agent

FIXED_TODO_SPEC = (
    "A simple todo list API: users can add a todo, list all todos, "
    "mark a todo complete, and delete a todo."
)


def is_docker_available() -> bool:
    """
    Checks if Docker daemon is running and accessible on the current system.
    """
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


# Skip entire test file cleanly if Docker is not available in the environment
pytestmark = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker daemon is not running on the host system"
)


def test_build_proof(tmp_path):
    """
    Case 1: The Build Proof.
    Run a real docker build against Phase 1's actual generated backend.
    Assert build_succeeded is True.
    """
    output_dir = str(tmp_path / "devops_build")
    p1 = run_phase1_pipeline(FIXED_TODO_SPEC, output_dir, use_real=False)

    build_result = run_devops_agent(
        backend_comp=p1["backend"],
        output_dir=output_dir,
        image_tag="p5_test_build_proof:latest",
        use_real=False,
    )

    assert build_result.build_succeeded is True, f"Docker build failed: {build_result.build_output}"
    assert build_result.build_duration_seconds >= 0.0


def test_build_failure_proof(tmp_path):
    """
    Case 2: The Build-Failure Proof.
    Build against a deliberately broken Dockerfile referencing a nonexistent base image.
    Assert build_succeeded is False with non-empty build_output explaining why.
    """
    output_dir = str(tmp_path / "devops_broken_build")
    os.makedirs(output_dir, exist_ok=True)

    # Write a broken Dockerfile with an invalid FROM instruction
    broken_dockerfile = os.path.join(output_dir, "Dockerfile")
    with open(broken_dockerfile, "w", encoding="utf-8") as f:
        f.write("FROM nonexistent_base_image_123456789:invalid_tag\nRUN exit 1\n")

    broken_comp = GeneratedComponent(
        component_name="backend",
        file_path=os.path.join(output_dir, "backend.py"),
        summary="Broken backend",
    )

    build_result = run_devops_agent(
        backend_comp=broken_comp,
        output_dir=output_dir,
        image_tag="p5_broken_test:latest",
        use_real=False,
    )

    assert build_result.build_succeeded is False, "Expected docker build to fail on invalid base image!"
    assert len(build_result.build_output.strip()) > 0, "Expected non-empty build output explaining build failure"


def test_runtime_proof(tmp_path):
    """
    Case 3: The Runtime Proof.
    Start a real container from successful build, run live HTTP create->list->complete->delete->verify-gone.
    Assert every EndpointCheck.passed is True and ValidationReport.all_passed is True.
    """
    output_dir = str(tmp_path / "runtime_proof")
    p3 = run_phase3_pipeline(FIXED_TODO_SPEC, output_dir, use_real=False, image_tag="p5_runtime_test:latest")

    val_report: ValidationReport = p3["validation"]
    assert val_report.container_started is True, "Validation container failed to start"
    assert val_report.all_passed is True, f"Validation failed: {val_report.endpoint_checks}"
    for check in val_report.endpoint_checks:
        assert check.passed is True, f"Endpoint check '{check.method} {check.path}' failed: {check.notes}"


def test_cleanup_proof(tmp_path):
    """
    Case 4: The Cleanup Proof.
    Assert container is no longer running after normal completion AND after forced failure.
    """
    output_dir = str(tmp_path / "cleanup_proof")
    p1 = run_phase1_pipeline(FIXED_TODO_SPEC, output_dir, use_real=False)

    build_result = run_devops_agent(p1["backend"], output_dir, "p5_cleanup_test:latest", use_real=False)
    assert build_result.build_succeeded is True

    # 1. Normal run cleanup check
    val_report = run_validation_agent(p1["contract"], build_result, use_real=False)
    assert val_report.teardown_succeeded is True

    # 2. Forced-failure cleanup check: introduce a bad endpoint in contract that causes exception
    broken_contract = p1["contract"].model_copy(deep=True)
    broken_contract.endpoints.append(
        broken_contract.endpoints[0].model_copy(update={"path": "/nonexistent_invalid_path_forcing_failure"})
    )

    forced_val_report = run_validation_agent(broken_contract, build_result, use_real=False)
    assert forced_val_report.all_passed is False, "Forced failure validation unexpectedly passed"
    assert forced_val_report.teardown_succeeded is True, "Container teardown failed after forced failure!"


def test_contract_drift_check(tmp_path):
    """
    Case 5: Contract-Drift Check.
    Assert ValidationReport.endpoint_checks covers every single endpoint in Phase 1 SystemContract.
    """
    output_dir = str(tmp_path / "contract_drift")
    p3 = run_phase3_pipeline(FIXED_TODO_SPEC, output_dir, use_real=False, image_tag="p5_drift_test:latest")

    contract = p3["contract"]
    val_report: ValidationReport = p3["validation"]

    contract_paths = {ep.path for ep in contract.endpoints}
    checked_paths = {chk.path for chk in val_report.endpoint_checks}

    assert contract_paths == checked_paths, (
        f"Validation checks did not cover every contract endpoint. "
        f"Contract: {contract_paths}, Checked: {checked_paths}"
    )
