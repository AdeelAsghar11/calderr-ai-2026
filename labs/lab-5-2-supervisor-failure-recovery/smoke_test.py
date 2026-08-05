"""
Smoke test for Week 5, Day 2 Supervisor Agent with Failure Recovery (supervisor_failure_recovery.py).
Runs offline with zero external API calls or credentials.

Validates:
1. Seeded deterministic cases:
   - Forced Timeout & Reroute: Specialist B times out, supervisor reroutes to another specialist.
   - Forced Low Confidence & Reroute: Specialist C returns low confidence, supervisor reroutes.
   - Forced Pool Exhaustion & Graceful Degradation: All specialists fail for a subtask, subtask degrades gracefully.
2. Stress pass:
   - Runs 40 random iterations end-to-end.
   - Asserts zero unhandled exceptions and exactly 3 subtask results per run.

Run:
    uv run python labs/lab-5-2-supervisor-failure-recovery/smoke_test.py
"""

from __future__ import annotations

import os
import random
import sys

# Ensure current directory is in Python path for direct script import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# pyrefly: ignore [missing-import]
from supervisor_failure_recovery import (
    ALL_SPECIALISTS,
    DelegationDecision,
    SpecialistResult,
    SpecialistRunner,
    SupervisorReport,
    build_graph,
    make_stub_runner,
)


def test_seeded_timeout_and_reroute():
    print("--- Test 1: Forced Timeout & Reroute ---")
    # Specialist B forced to time out (b_timeout_prob=1.0), Specialist C reliable (c_low_conf_prob=0.0)
    runner = make_stub_runner(rng=random.Random(42), b_timeout_prob=1.0, c_low_conf_prob=0.0)
    graph = build_graph(runner=runner)

    result = graph.invoke({
        "original_task": "Build robust payment gateway",
        "log": [],
        "subtask_results": [],
        "degraded_subtasks": [],
        "delegation_log": [],
    })

    report: SupervisorReport = result["report"]
    assert report.original_task == "Build robust payment gateway"
    assert len(report.subtask_results) == 3, f"Expected 3 subtask results, got {len(report.subtask_results)}"

    # Verify Specialist B timed out and decision was logged with reasoning
    timeout_decisions = [d for d in report.delegation_log if d.outcome == "timeout"]
    assert len(timeout_decisions) >= 1, "Expected at least one timeout decision logged"

    first_timeout = timeout_decisions[0]
    assert first_timeout.specialist_name == "Specialist B"
    assert "timed out" in first_timeout.reasoning.lower()
    assert first_timeout.reasoning != ""

    # Verify subtask 2 (primary Specialist B) was rerouted to alternative specialist and succeeded
    subtask_2_res = report.subtask_results[1]
    assert subtask_2_res.succeeded, "Subtask 2 should succeed via rerouting to alternative specialist"
    assert subtask_2_res.specialist_name != "Specialist B", "Subtask 2 should be resolved by alternative specialist"
    print(f"  Passed: Specialist B timed out on attempt #1, rerouted to '{subtask_2_res.specialist_name}' and succeeded.")


def test_seeded_low_confidence_and_reroute():
    print("\n--- Test 2: Forced Low Confidence & Reroute ---")
    # Specialist C forced to return low confidence (c_low_conf_prob=1.0), Specialist B no timeout (b_timeout_prob=0.0)
    runner = make_stub_runner(rng=random.Random(123), b_timeout_prob=0.0, c_low_conf_prob=1.0)
    graph = build_graph(runner=runner)

    result = graph.invoke({
        "original_task": "Implement data pipeline",
        "log": [],
        "subtask_results": [],
        "degraded_subtasks": [],
        "delegation_log": [],
    })

    report: SupervisorReport = result["report"]
    assert len(report.subtask_results) == 3

    low_conf_decisions = [d for d in report.delegation_log if d.outcome == "low_confidence"]
    assert len(low_conf_decisions) >= 1, "Expected at least one low_confidence decision"

    first_low_conf = low_conf_decisions[0]
    assert first_low_conf.specialist_name == "Specialist C"
    assert "low confidence" in first_low_conf.reasoning.lower()

    # Subtask 3 (primary Specialist C) should have rerouted and succeeded
    subtask_3_res = report.subtask_results[2]
    assert subtask_3_res.succeeded, "Subtask 3 should succeed via rerouting"
    assert subtask_3_res.specialist_name != "Specialist C"
    print(f"  Passed: Specialist C returned low confidence, rerouted to '{subtask_3_res.specialist_name}' and succeeded.")


def test_seeded_pool_exhaustion_and_degradation():
    print("\n--- Test 3: Forced Pool Exhaustion & Graceful Degradation ---")

    class ExhaustionRunner(SpecialistRunner):
        def run_specialist(self, specialist_name: str, subtask: str) -> SpecialistResult:
            if "Phase 2" in subtask:
                if specialist_name == "Specialist B":
                    raise TimeoutError("Specialist B timeout on Phase 2")
                return SpecialistResult(
                    subtask=subtask,
                    specialist_name=specialist_name,
                    content="Low confidence stub for Phase 2",
                    confidence=0.35,
                    succeeded=False,
                )
            return super().run_specialist(specialist_name, subtask)

    runner = ExhaustionRunner(rng=random.Random(999), b_timeout_prob=0.0, c_low_conf_prob=0.0)
    graph = build_graph(runner=runner)

    result = graph.invoke({
        "original_task": "Mission critical deployment",
        "log": [],
        "subtask_results": [],
        "degraded_subtasks": [],
        "delegation_log": [],
    })

    report: SupervisorReport = result["report"]
    assert len(report.subtask_results) == 3
    assert len(report.degraded_subtasks) == 1
    assert report.overall_status == "degraded"

    exhausted_decisions = [d for d in report.delegation_log if d.outcome == "exhausted"]
    assert len(exhausted_decisions) == 1
    assert exhausted_decisions[0].attempt_number == 3

    degraded_result = report.subtask_results[1]
    assert not degraded_result.succeeded
    assert "[DEGRADED]" in degraded_result.content
    print("  Passed: All 3 specialists failed for Phase 2, subtask gracefully degraded without throwing an exception.")


def test_stress_pass():
    print("\n--- Test 4: Stress Pass (40 Random Runs) ---")
    num_runs = 40
    completed_count = 0
    degraded_count = 0

    for i in range(num_runs):
        seed = 2000 + i
        runner = make_stub_runner(rng=random.Random(seed), b_timeout_prob=0.5, c_low_conf_prob=0.5)
        graph = build_graph(runner=runner)

        try:
            result = graph.invoke({
                "original_task": f"Stress test run #{i+1}",
                "log": [],
                "subtask_results": [],
                "degraded_subtasks": [],
                "delegation_log": [],
            })
        except Exception as err:
            assert False, f"Run #{i+1} (seed={seed}) raised unhandled exception: {err}"

        report: SupervisorReport = result["report"]
        assert len(report.subtask_results) == 3, f"Run #{i+1}: expected 3 subtask results, got {len(report.subtask_results)}"
        assert len(report.delegation_log) >= 3, f"Run #{i+1}: expected at least 3 delegation decisions"
        assert report.overall_status in ("complete", "degraded")

        if report.overall_status == "complete":
            completed_count += 1
        else:
            degraded_count += 1

    print(f"  Passed: {num_runs} random stress runs completed with ZERO unhandled exceptions.")
    print(f"          Status Breakdown: {completed_count} completed fully, {degraded_count} degraded gracefully.")


if __name__ == "__main__":
    print("=== Running Lab 5.2 Supervisor Failure Recovery Smoke Tests ===")
    test_seeded_timeout_and_reroute()
    test_seeded_low_confidence_and_reroute()
    test_seeded_pool_exhaustion_and_degradation()
    test_stress_pass()
    print("\nALL SMOKE TESTS PASSED CLEANLY!")
