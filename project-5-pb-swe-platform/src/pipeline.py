"""
Multi-agent software engineering pipeline orchestrator.

Assembles PM Agent, Architect Agent, Backend Agent, Frontend Agent, Integration Agent,
QA Agent, Security Agent, DevOps Agent, and Validation Agent into a multi-agent execution pipeline.
"""

from typing import Any, Dict
from src.architect_agent import run_architect_agent
from src.backend_agent import run_backend_agent
from src.devops_agent import run_devops_agent
from src.frontend_agent import run_frontend_agent
from src.integration_agent import run_integration_agent
from src.pm_agent import run_pm_agent
from src.qa_agent import run_qa_agent
from src.security_agent import run_security_agent
from src.validation_agent import run_validation_agent


def run_phase1_pipeline(
    feature_spec: str,
    output_dir: str,
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Runs the Phase 1 pipeline end-to-end (PM -> Architect -> Backend + Frontend -> Integration).
    Maintained for backward compatibility with Phase 1 verification tests.
    """
    task_list = run_pm_agent(feature_spec=feature_spec, use_real=use_real)
    contract = run_architect_agent(task_list=task_list, use_real=use_real)
    backend_comp = run_backend_agent(contract=contract, output_dir=output_dir, use_real=use_real)
    frontend_comp = run_frontend_agent(contract=contract, output_dir=output_dir, use_real=use_real)
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


def run_phase2_pipeline(
    feature_spec: str,
    output_dir: str,
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Runs the complete Phase 2 pipeline end-to-end (PM -> Architect -> Backend/Frontend -> Integration -> QA -> Security).
    """
    p1_results = run_phase1_pipeline(feature_spec=feature_spec, output_dir=output_dir, use_real=use_real)

    qa_report = run_qa_agent(
        task_list=p1_results["task_list"],
        backend_comp=p1_results["backend"],
        output_dir=output_dir,
        use_real=use_real,
    )

    security_report = run_security_agent(
        backend_comp=p1_results["backend"],
        frontend_comp=p1_results["frontend"],
        use_real=use_real,
    )

    results = dict(p1_results)
    results["qa"] = qa_report
    results["security"] = security_report
    return results


def run_phase3_pipeline(
    feature_spec: str,
    output_dir: str,
    use_real: bool = False,
    image_tag: str = "project5_backend:latest",
) -> Dict[str, Any]:
    """
    Runs the complete Phase 3 pipeline end-to-end.

    Flow:
    1. PM Agent decomposes feature spec into structured TaskList.
    2. Architect Agent designs typed SystemContract.
    3. Backend Agent writes FastAPI application file to disk based on SystemContract.
    4. Frontend Agent writes HTML/JS web application file to disk based on SystemContract.
    5. Integration Agent verifies generated files directly against SystemContract endpoints.
    6. QA Agent generates and executes pytest suite in a sandboxed subprocess.
    7. Security Agent performs static analysis scanning for OWASP Top 10 security findings.
    8. DevOps Agent writes Dockerfile/Compose and executes docker build.
    9. Validation Agent starts live container on dynamic port, verifies HTTP endpoints, and tears down container.
    """
    p2_results = run_phase2_pipeline(feature_spec=feature_spec, output_dir=output_dir, use_real=use_real)

    # Step 8: DevOps Agent
    devops_result = run_devops_agent(
        backend_comp=p2_results["backend"],
        output_dir=output_dir,
        image_tag=image_tag,
        use_real=use_real,
    )

    # Step 9: Validation Agent
    validation_report = run_validation_agent(
        contract=p2_results["contract"],
        build_result=devops_result,
        use_real=use_real,
    )

    results = dict(p2_results)
    results["devops"] = devops_result
    results["validation"] = validation_report
    return results
