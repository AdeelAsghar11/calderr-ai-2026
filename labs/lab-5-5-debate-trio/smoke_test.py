"""
Smoke test for lab-5-5-debate-trio (debate_trio.py).

Tests offline (stub mode):
1. Straightforward case: Proposer statements are substantively stronger throughout.
   Asserts the Arbiter sides with the Proposer.
2. Recency-bias proof:
   - Part A: Strong early Proposer statement (Round 0/2) vs generic final Challenger statement (Round 3).
     Asserts Arbiter sides with Proposer (earlier strong argument beats later weak argument).
   - Part B (Mirrored proof): Strong Challenger statement (Round 1/3) vs generic Proposer statements.
     Asserts Arbiter verdict flips to Challenger (proving position-independence and role-independence).

Run:
    uv run python labs/lab-5-5-debate-trio/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure lab directory is in sys.path for clean imports
lab_dir = Path(__file__).parent
if str(lab_dir) not in sys.path:
    sys.path.insert(0, str(lab_dir))

from debate_trio import (
    ArbiterVerdict,
    DebateTranscript,
    Statement,
    run_debate,
    stub_arbiter_eval,
)


def test_straightforward_case():
    print("\n--- Test Case 1: Straightforward Debate (Proposer Stronger) ---")
    question = "Should we adopt automated CI/CD pipeline tests?"

    def proposer_provider(q: str, round_num: int, prev: list[Statement]) -> str:
        if round_num == 0:
            return (
                "Proposer Round 0: Automated CI/CD pipelines reduce deployment failure rates by 70%, "
                "providing concrete test coverage metrics, latency profiling, and schema invariant validation."
            )
        return (
            "Proposer Round 2: Specifically addressing the challenger, automated refactoring and compile-time contracts "
            "eliminate regression bottlenecks across all microservice deployments."
        )

    def challenger_provider(q: str, round_num: int, prev: list[Statement]) -> str:
        if round_num == 1:
            return "Challenger Round 1: Setup takes time."
        return "Challenger Round 3: Generic statement, i just disagree."

    transcript, verdict = run_debate(
        question,
        is_real=False,
        proposer_provider=proposer_provider,
        challenger_provider=challenger_provider,
    )

    print(f"  Question:         '{question}'")
    print(f"  Winning Side:     '{verdict.winning_side}'")
    print(f"  Decisive Round:   Round {verdict.decisive_round}")
    print(f"  Reasoning:        '{verdict.reasoning}'")

    assert isinstance(verdict, ArbiterVerdict)
    assert verdict.winning_side == "proposer", f"Expected proposer to win, got {verdict.winning_side}"
    assert verdict.decisive_round in [0, 2], f"Expected decisive round 0 or 2, got {verdict.decisive_round}"
    print("Straightforward case assertions passed clean.")


def test_recency_bias_proof():
    print("\n--- Test Case 2: Recency-Bias Proof (Anti-Recency & Mirrored Flipping) ---")
    question = "Should early-stage startups use microservices?"

    print("\n[Part 2A: Strong Early Proposer (Round 0/2) vs Weak Final Challenger (Round 3)]")
    # Strong Proposer statement in early round (Round 0), weak final statement from Challenger (Round 3)
    statements_2a = [
        Statement(
            round=0,
            role="proposer",
            content=(
                "Proposer Round 0 (Strong Early Argument): Monoliths excel early because microservices "
                "introduce massive operational overhead, distributed transaction latency, network failure modes, "
                "and complex schema coupling without clear throughput benchmarks."
            ),
        ),
        Statement(
            round=1,
            role="challenger",
            content="Challenger Round 1: Microservices allow independent teams.",
        ),
        Statement(
            round=2,
            role="proposer",
            content="Proposer Round 2: Specifically, team headcount under 10 does not justify network boundary latency.",
        ),
        Statement(
            round=3,
            role="challenger",
            content="Challenger Round 3 (Weak Final Statement): Generic statement, whatever, my position is obvious, i just disagree.",
        ),
    ]

    transcript_2a = DebateTranscript(question=question, statements=statements_2a)
    verdict_2a = stub_arbiter_eval(transcript_2a)

    print(f"  Part 2A Winner:   '{verdict_2a.winning_side}'")
    print(f"  Decisive Round:   Round {verdict_2a.decisive_round}")
    print(f"  Reasoning:        '{verdict_2a.reasoning}'")

    assert verdict_2a.winning_side == "proposer", (
        f"Recency-bias defect! Final statement (Round 3 Challenger) won despite being generic filler. "
        f"Got winning_side='{verdict_2a.winning_side}'."
    )
    assert verdict_2a.decisive_round == 0, f"Expected Round 0 to be decisive, got {verdict_2a.decisive_round}"
    print("Part 2A passed: Strong earlier statement beats weak final statement.")

    print("\n[Part 2B: Mirrored Check -- Strong Challenger (Round 1/3) vs Weak Proposer]")
    # Swap content: Challenger has the strong detailed argument, Proposer has weak generic statements
    statements_2b = [
        Statement(
            round=0,
            role="proposer",
            content="Proposer Round 0 (Weak): Generic statement, my position is obvious, i just disagree.",
        ),
        Statement(
            round=1,
            role="challenger",
            content=(
                "Challenger Round 1 (Strong Argument): Examining the claim specifically, serverless edge functions "
                "reduce latency bottlenecks, provide isolated failure modes, dynamic schema safety, and concrete benchmark metrics."
            ),
        ),
        Statement(
            round=2,
            role="proposer",
            content="Proposer Round 2 (Weak): Whatever, no comment as i said before.",
        ),
        Statement(
            round=3,
            role="challenger",
            content="Challenger Round 3: Specifically disproving the proposer, operational overhead is automated via CI/CD.",
        ),
    ]

    transcript_2b = DebateTranscript(question=question, statements=statements_2b)
    verdict_2b = stub_arbiter_eval(transcript_2b)

    print(f"  Part 2B Winner:   '{verdict_2b.winning_side}'")
    print(f"  Decisive Round:   Round {verdict_2b.decisive_round}")
    print(f"  Reasoning:        '{verdict_2b.reasoning}'")

    assert verdict_2b.winning_side == "challenger", (
        f"Role bias defect! Swapped content did not flip verdict. Expected 'challenger', got '{verdict_2b.winning_side}'."
    )
    assert verdict_2b.decisive_round == 1, f"Expected Round 1 to be decisive, got {verdict_2b.decisive_round}"
    print("Part 2B passed: Mirrored check flipped verdict to Challenger as expected!")
    print("\nPROVED: Arbiter evaluates content quality independently of statement recency or fixed role bias!")


if __name__ == "__main__":
    test_straightforward_case()
    test_recency_bias_proof()
