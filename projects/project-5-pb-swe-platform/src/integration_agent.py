"""
Integration Agent implementation.

Performs deterministic, non-generative verification of generated Backend and Frontend code artifacts
against the authoritative SystemContract.
Acts as an automated compliance auditor to verify cross-component contract consistency.
"""

import os
from src.schema import GeneratedComponent, IntegrationResult, SystemContract


def run_integration_agent(
    contract: SystemContract,
    backend_comp: GeneratedComponent,
    frontend_comp: GeneratedComponent,
) -> IntegrationResult:
    """
    Verifies that generated backend and frontend source files both implement every endpoint path specified in SystemContract.

    Why verification is strictly non-generative:
    Verification must be an objective check against the source-of-truth contract.
    Allowing verification to generate or modify code would introduce non-deterministic bias.
    """
    if not os.path.exists(backend_comp.file_path):
        raise FileNotFoundError(f"Backend component file not found at: {backend_comp.file_path}")
    if not os.path.exists(frontend_comp.file_path):
        raise FileNotFoundError(f"Frontend component file not found at: {frontend_comp.file_path}")

    with open(backend_comp.file_path, "r", encoding="utf-8") as f:
        backend_content = f.read()

    with open(frontend_comp.file_path, "r", encoding="utf-8") as f:
        frontend_content = f.read()

    missing_in_backend: list[str] = []
    missing_in_frontend: list[str] = []

    for endpoint in contract.endpoints:
        path = endpoint.path

        # Check backend presence: route decorator or path string must exist
        if path not in backend_content:
            missing_in_backend.append(path)

        # Check frontend presence: fetch call or path string must exist
        if path not in frontend_content:
            missing_in_frontend.append(path)

    consistent = len(missing_in_backend) == 0 and len(missing_in_frontend) == 0

    if consistent:
        notes = (
            f"Successfully verified contract '{contract.feature_name}': "
            f"All {len(contract.endpoints)} endpoints exist in both backend and frontend artifacts."
        )
    else:
        notes = (
            f"Contract mismatch detected for feature '{contract.feature_name}'. "
            f"Missing in backend: {missing_in_backend}; Missing in frontend: {missing_in_frontend}."
        )

    return IntegrationResult(
        consistent=consistent,
        missing_in_backend=missing_in_backend,
        missing_in_frontend=missing_in_frontend,
        notes=notes,
    )
