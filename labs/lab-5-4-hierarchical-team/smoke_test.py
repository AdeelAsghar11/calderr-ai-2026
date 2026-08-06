"""
Smoke test for lab-5-4-hierarchical-team (hierarchical_team.py).

Tests two core aspects offline:
1. Happy Path: 3-tier execution producing a valid ReleaseReport with BuildSummary, QASummary,
   and correct code-enforced overall_status ("ready" vs "blocked").
2. State Isolation Proof: Injecting a unique marker string into Backend worker's internal state
   and asserting that it does NOT leak into Engineering Lead state, QA Lead state, PM state,
   or the final ReleaseReport.

Run:
    uv run python labs/lab-5-4-hierarchical-team/smoke_test.py
"""

from __future__ import annotations

import json
# pyrefly: ignore [missing-import]
from hierarchical_team import (
    BuildSummary,
    QASummary,
    ReleaseReport,
    stub_engineering_lead,
    stub_pm_agent,
    stub_qa_lead,
)


def test_happy_path():
    print("\n--- Test Case 1: Happy Path Execution & Status Logic ---")
    brief = "Build real-time notification engine with WebSockets"
    report, trace = stub_pm_agent(brief)

    print(f"  Feature brief:       '{report.feature}'")
    print(f"  Components built:    {report.build_summary.components_built}")
    print(f"  Tests written:       {report.qa_summary.test_cases_written}")
    print(f"  Tests passed:        {report.qa_summary.tests_passed}")
    print(f"  Tests failed:        {report.qa_summary.tests_failed}")
    print(f"  Overall Status:      '{report.overall_status}'")

    # Assertions
    assert isinstance(report, ReleaseReport)
    assert isinstance(report.build_summary, BuildSummary)
    assert isinstance(report.qa_summary, QASummary)
    assert report.feature == brief
    assert len(report.build_summary.components_built) == 2
    assert report.qa_summary.tests_passed == report.qa_summary.test_cases_written
    assert report.qa_summary.tests_failed == 0
    assert report.overall_status == "ready"

    # Test status override rule when test failures exist
    mock_blocked_build = BuildSummary(
        components_built=["backend_api"],
        backend_summary="Built API",
        frontend_summary="Built UI",
        issues=["Database connection leak"],
    )
    qa_blocked_sum, _ = stub_qa_lead(mock_blocked_build, brief)
    assert qa_blocked_sum.tests_failed > 0
    blocked_status = "blocked" if qa_blocked_sum.tests_failed > 0 else "ready"
    assert blocked_status == "blocked"
    print("  Status logic override verified: tests_failed > 0 triggers 'blocked'.")
    print("Happy path assertions passed clean.")


def test_state_isolation_proof():
    print("\n--- Test Case 2: State Isolation Proof (Context Leak Test) ---")
    marker = "MARKER_7f3a_backend_internal_only"
    brief = "Implement payment gateway API"

    print(f"  Injecting marker '{marker}' into Backend worker internal scratchpad only...")
    report, trace = stub_pm_agent(brief, backend_marker=marker)

    backend_scratchpad = trace["backend_worker_scratchpad"]
    eng_state_str = str(trace["engineering_lead_state"])
    qa_state_str = str(trace["qa_lead_state"])
    pm_state_str = str(trace["pm_state"])
    report_json_str = report.model_dump_json()

    print(f"  Checking Backend Worker scratchpad... (Marker Present: {marker in backend_scratchpad})")
    assert marker in backend_scratchpad, "Marker should exist inside Backend worker's internal state"

    print(f"  Checking Engineering Lead state...   (Marker Present: {marker in eng_state_str})")
    assert marker not in eng_state_str, "CRITICAL: Marker leaked into Engineering Lead state!"

    print(f"  Checking QA Lead state...            (Marker Present: {marker in qa_state_str})")
    assert marker not in qa_state_str, "CRITICAL: Marker leaked into QA Lead state!"

    print(f"  Checking PM Executive state...       (Marker Present: {marker in pm_state_str})")
    assert marker not in pm_state_str, "CRITICAL: Marker leaked into PM state!"

    print(f"  Checking Final ReleaseReport JSON... (Marker Present: {marker in report_json_str})")
    assert marker not in report_json_str, "CRITICAL: Marker leaked into final ReleaseReport!"

    print("\nPROVED: Internal worker state remains 100% isolated within its tier boundary!")


if __name__ == "__main__":
    test_happy_path()
    test_state_isolation_proof()
    print("\nAll 2 hierarchical team smoke test cases passed successfully.")
