"""
Multi-agent software engineering pipeline orchestrator.

Assembles PM Agent, Architect Agent, Backend Agent, Frontend Agent, Integration Agent,
QA Agent, Security Agent, DevOps Agent, and Validation Agent into a complete multi-agent execution pipeline.
Supports graceful handling of optional Docker stages when Docker daemon is unavailable.
"""

import uuid
from typing import Any, Dict, Optional
from src.architect_agent import run_architect_agent
from src.backend_agent import run_backend_agent
from src.devops_agent import run_devops_agent
from src.frontend_agent import run_frontend_agent
from src.integration_agent import run_integration_agent
from src.pm_agent import run_pm_agent
from src.qa_agent import run_qa_agent
from src.schema import PipelineRun
from src.security_agent import run_security_agent
from src.validation_agent import _is_container_running, is_docker_available, run_validation_agent


def run_phase1_pipeline(
    feature_spec: str,
    output_dir: str,
    use_real: bool = False,
) -> Dict[str, Any]:
    """Phase 1 pipeline backward compatibility wrapper."""
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
    """Phase 2 pipeline backward compatibility wrapper."""
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
    """Phase 3 pipeline backward compatibility wrapper."""
    p2_results = run_phase2_pipeline(feature_spec=feature_spec, output_dir=output_dir, use_real=use_real)
    devops_result = run_devops_agent(
        backend_comp=p2_results["backend"],
        output_dir=output_dir,
        image_tag=image_tag,
        use_real=use_real,
    )
    validation_report = run_validation_agent(
        contract=p2_results["contract"],
        build_result=devops_result,
        use_real=use_real,
    )
    results = dict(p2_results)
    results["devops"] = devops_result
    results["validation"] = validation_report
    return results


def run_full_pipeline(
    feature_spec: str,
    output_dir: str,
    use_real: bool = False,
    run_id: Optional[str] = None,
    image_tag: str = "project5_backend:latest",
) -> PipelineRun:
    """
    Executes the complete multi-agent pipeline and returns a structured PipelineRun artifact.

    Why stages_completed and optional Docker fields are handled explicitly:
    If Docker is not running in the current host environment, skipping DevOps and Validation stages
    should produce a clean completed pipeline run (with docker_build_result=None and validation_report=None)
    rather than causing the overall pipeline to fail.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    stages_completed: list[str] = []

    # Step 1: PM Agent
    task_list = run_pm_agent(feature_spec=feature_spec, use_real=use_real)
    stages_completed.append("pm")

    # Step 2: Architect Agent
    contract = run_architect_agent(task_list=task_list, use_real=use_real)
    stages_completed.append("architect")

    # Step 3: Backend Agent
    backend_comp = run_backend_agent(contract=contract, output_dir=output_dir, use_real=use_real)
    stages_completed.append("backend")

    # Step 4: Frontend Agent
    frontend_comp = run_frontend_agent(contract=contract, output_dir=output_dir, use_real=use_real)
    stages_completed.append("frontend")

    # Step 5: Integration Agent
    integration_result = run_integration_agent(
        contract=contract,
        backend_comp=backend_comp,
        frontend_comp=frontend_comp,
    )
    stages_completed.append("integration")

    # Step 6: QA Agent
    qa_report = run_qa_agent(
        task_list=task_list,
        backend_comp=backend_comp,
        output_dir=output_dir,
        use_real=use_real,
    )
    stages_completed.append("qa")

    # Step 7: Security Agent
    security_report = run_security_agent(
        backend_comp=backend_comp,
        frontend_comp=frontend_comp,
        use_real=use_real,
    )
    stages_completed.append("security")

    # Step 8 & 9: DevOps & Validation Agents (Conditional on Docker availability)
    docker_build_result = None
    validation_report = None

    if is_docker_available():
        docker_build_result = run_devops_agent(
            backend_comp=backend_comp,
            output_dir=output_dir,
            image_tag=image_tag,
            use_real=use_real,
        )
        stages_completed.append("devops")

        validation_report = run_validation_agent(
            contract=contract,
            build_result=docker_build_result,
            use_real=use_real,
        )
        stages_completed.append("validation")

    return PipelineRun(
        run_id=run_id,
        feature_spec=feature_spec,
        status="completed",
        stages_completed=stages_completed,
        task_list=task_list,
        system_contract=contract,
        qa_report=qa_report,
        security_report=security_report,
        docker_build_result=docker_build_result,
        validation_report=validation_report,
    )
