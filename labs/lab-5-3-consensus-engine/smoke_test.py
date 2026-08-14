"""
Smoke test for lab-5-3-consensus-engine (consensus_engine.py).

Tests offline (stub mode):
1. Clean consensus: All specialists agree in Round 1.
   Asserts rounds_used == 1, cleared_threshold is True, dissent_summary == "".
2. Escalation proof: Round 1 splits with no verdict clearing 60%.
   Asserts top-2 specialists by confidence are selected, Round 2 runs with top 2,
   the lowest-confidence specialist is excluded from Round 2 evaluation, and rounds_used == 2.
3. Never-crash proof: Even Round 2 does not clear 60%.
   Asserts returns ConsensusVerdict with cleared_threshold is False without looping past Round 2 or throwing.

Run:
    uv run python labs/lab-5-3-consensus-engine/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure lab directory is in sys.path for clean imports
lab_dir = Path(__file__).parent
if str(lab_dir) not in sys.path:
    sys.path.insert(0, str(lab_dir))

# pyrefly: ignore [missing-import]
from consensus_engine import (
    ConsensusVerdict,
    SpecialistOpinion,
    run_consensus,
)


def test_clean_consensus():
    print("\n--- Test Case 1: Clean Consensus (Round 1 Agreement) ---")
    clean_code = """
def calculate_area(length: float, width: float) -> float:
    \"\"\"Calculates rectangle area with clean type hints and docstring.\"\"\"
    return length * width
"""

    opinions, verdict = run_consensus(clean_code, is_real=False)

    print(f"  Final Verdict:               '{verdict.final_verdict}'")
    print(f"  Weighted Confidence Share:  {verdict.weighted_confidence_share:.2%}")
    print(f"  Rounds Used:                {verdict.rounds_used}")
    print(f"  Cleared Threshold:          {verdict.cleared_threshold}")
    print(f"  Dissent Summary:            '{verdict.dissent_summary}'")

    assert isinstance(verdict, ConsensusVerdict)
    assert verdict.final_verdict == "approve"
    assert verdict.weighted_confidence_share >= 0.60
    assert verdict.rounds_used == 1
    assert verdict.cleared_threshold is True
    assert verdict.dissent_summary == "", "Clean consensus must yield empty dissent_summary"
    print("Clean consensus assertions passed clean.")


def test_escalation_proof():
    print("\n--- Test Case 2: Escalation Proof (Split Round 1 -> Top-2 Round 2) ---")
    split_code = "# Code under review for split test"

    # Custom providers to construct exact confidence split in Round 1:
    # Security: approve (conf = 0.50) -> Top 1
    # Performance: needs_changes (conf = 0.40) -> Top 2
    # Maintainability: reject (conf = 0.10) -> Excluded (Lowest confidence)
    # Total conf = 1.0. Leading share = 0.50 / 1.00 = 50% < 60%. Escalation triggered!

    def sec_provider(code: str) -> SpecialistOpinion:
        return SpecialistOpinion(
            specialist_name="Security Agent",
            verdict="approve",
            confidence=0.50,
            reasoning="Security looks acceptable.",
            findings=["Clean auth."],
        )

    def perf_provider(code: str) -> SpecialistOpinion:
        return SpecialistOpinion(
            specialist_name="Performance Agent",
            verdict="needs_changes",
            confidence=0.40,
            reasoning="Performance needs optimization.",
            findings=["Loop optimization required."],
        )

    def maint_provider(code: str) -> SpecialistOpinion:
        return SpecialistOpinion(
            specialist_name="Maintainability Agent",
            verdict="reject",
            confidence=0.10,
            reasoning="Unmaintainable code.",
            findings=["Spaghetti code."],
        )

    # In Round 2, only Security (approve) and Performance (needs_changes) should run.
    # We supply a custom Round 2 provider where Security shifts to needs_changes (conf 0.80)
    # and Performance states needs_changes (conf 0.90).
    # Result in Round 2: needs_changes = 100% share >= 60%.
    def round_2_override(code: str, top_2_names: list[str], summary: str) -> list[SpecialistOpinion]:
        assert "Maintainability Agent" not in top_2_names, (
            f"CRITICAL ERROR: Lowest confidence specialist 'Maintainability Agent' was NOT excluded! Top 2: {top_2_names}"
        )
        assert len(top_2_names) == 2
        return [
            SpecialistOpinion(
                specialist_name="Security Agent",
                verdict="needs_changes",
                confidence=0.80,
                reasoning="Agreed with performance concerns in Round 2.",
                findings=["Needs optimization."],
            ),
            SpecialistOpinion(
                specialist_name="Performance Agent",
                verdict="needs_changes",
                confidence=0.90,
                reasoning="Re-affirming needs_changes.",
                findings=["Loop latency."],
            ),
        ]

    r1_opinions, verdict = run_consensus(
        split_code,
        is_real=False,
        security_provider=sec_provider,
        performance_provider=perf_provider,
        maintainability_provider=maint_provider,
        round_2_override_provider=round_2_override,
    )

    print(f"  Final Verdict:               '{verdict.final_verdict}'")
    print(f"  Weighted Confidence Share:  {verdict.weighted_confidence_share:.2%}")
    print(f"  Rounds Used:                {verdict.rounds_used}")
    print(f"  Cleared Threshold:          {verdict.cleared_threshold}")
    print(f"  Dissent Summary:            '{verdict.dissent_summary[:60]}...'")

    assert verdict.rounds_used == 2, f"Expected rounds_used == 2, got {verdict.rounds_used}"
    assert verdict.cleared_threshold is True
    assert verdict.final_verdict == "needs_changes"
    assert verdict.dissent_summary != "", "Split round 1 must populate dissent_summary"

    # Verify excluded specialist's original verdict ('reject') was not used
    assert verdict.final_verdict != "reject", "Excluded specialist's verdict leaked into final result!"
    print("Escalation proof passed: Top 2 correctly isolated and executed in Round 2.")


def test_never_crash_proof():
    print("\n--- Test Case 3: Never-Crash Proof (Round 2 Share < 60% Cap) ---")
    code = "# Code that remains split after Round 2"

    # Round 1 split:
    # Security: approve (conf 0.50)
    # Performance: reject (conf 0.40)
    # Maintainability: needs_changes (conf 0.10) -> Excluded
    def sec_provider(code: str) -> SpecialistOpinion:
        return SpecialistOpinion(
            specialist_name="Security Agent",
            verdict="approve",
            confidence=0.50,
            reasoning="Approve.",
            findings=[],
        )

    def perf_provider(code: str) -> SpecialistOpinion:
        return SpecialistOpinion(
            specialist_name="Performance Agent",
            verdict="reject",
            confidence=0.40,
            reasoning="Reject.",
            findings=[],
        )

    def maint_provider(code: str) -> SpecialistOpinion:
        return SpecialistOpinion(
            specialist_name="Maintainability Agent",
            verdict="needs_changes",
            confidence=0.10,
            reasoning="Needs changes.",
            findings=[],
        )

    # Round 2 also remains split below 60%:
    # Security: approve (conf 0.51) -> 51% share < 60%
    # Performance: reject (conf 0.49) -> 49% share
    def round_2_override(code: str, top_2_names: list[str], summary: str) -> list[SpecialistOpinion]:
        return [
            SpecialistOpinion(
                specialist_name="Security Agent",
                verdict="approve",
                confidence=0.51,
                reasoning="Still approve.",
                findings=[],
            ),
            SpecialistOpinion(
                specialist_name="Performance Agent",
                verdict="reject",
                confidence=0.49,
                reasoning="Still reject.",
                findings=[],
            ),
        ]

    r1_opinions, verdict = run_consensus(
        code,
        is_real=False,
        security_provider=sec_provider,
        performance_provider=perf_provider,
        maintainability_provider=maint_provider,
        round_2_override_provider=round_2_override,
    )

    print(f"  Final Verdict:               '{verdict.final_verdict}'")
    print(f"  Weighted Confidence Share:  {verdict.weighted_confidence_share:.2%}")
    print(f"  Rounds Used:                {verdict.rounds_used}")
    print(f"  Cleared Threshold:          {verdict.cleared_threshold}")

    assert isinstance(verdict, ConsensusVerdict)
    assert verdict.rounds_used == 2, f"System must stop at max 2 rounds, got {verdict.rounds_used}"
    assert verdict.cleared_threshold is False, "cleared_threshold must be False when share < 0.60"
    assert verdict.final_verdict == "approve", "Should pick leading verdict (approve at 51%)"
    print("Never-crash proof passed: Engine returned marked low-confidence verdict without looping or throwing.")


if __name__ == "__main__":
    test_clean_consensus()
    test_escalation_proof()
    test_never_crash_proof()
