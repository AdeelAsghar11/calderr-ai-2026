"""
Phase 1 pipeline orchestrator.

Assembles PM Agent, Architect Agent, Backend Agent, Frontend Agent, and Integration Agent
into a sequential multi-agent execution pipeline.
"""

from typing import Any, Dict
from src.architect_agent import run_architect_agent
from src.backend_agent import run_backend_agent
from src.frontend_agent import run_frontend_agent
from src.integration_agent import run_integration_agent
from src.pm_agent import run_pm_agent


def run_phase1_pipeline(
    feature_spec: str,
    output_dir: str,
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Runs the complete Phase 1 pipeline end-to-end.

    Flow:
    1. PM Agent decomposes feature spec into structured TaskList with acceptance criteria.
    2. Architect Agent designs typed SystemContract from TaskList.
    3. Backend Agent writes FastAPI application file to disk based on SystemContract.
    4. Frontend Agent writes HTML/JS web application file to disk based on SystemContract.
    5. Integration Agent verifies generated files directly against SystemContract endpoints.

    Why this pipeline sequence is enforced:
    The contract must be established by the Architect before code generation starts, ensuring both
    Backend and Frontend agents generate against an identical specification boundary.
    """
    # Step 1: PM Agent
    task_list = run_pm_agent(feature_spec=feature_spec, use_real=use_real)

    # Step 2: Architect Agent
    contract = run_architect_agent(task_list=task_list, use_real=use_real)

    # Step 3: Backend Agent
    backend_comp = run_backend_agent(contract=contract, output_dir=output_dir, use_real=use_real)

    # Step 4: Frontend Agent
    frontend_comp = run_frontend_agent(contract=contract, output_dir=output_dir, use_real=use_real)

    # Step 5: Integration Agent
    integration_result = run_integration_agent(
        contract=contract,
        backend_comp=backend_comp,
        frontend_comp=frontend_comp,
    )

    return {
        "task_list": task_list,
        "contract": contract,
        "backend": backend_comp,
        "frontend": frontend_comp,
        "integration": integration_result,
    }
