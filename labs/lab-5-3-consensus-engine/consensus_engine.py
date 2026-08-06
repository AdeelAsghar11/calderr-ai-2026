r"""
Week 5, Day 4 (Thursday Core Learning) -- Lab 5.3: Confidence-Weighted Consensus Engine
CalderR Agentic AI Engineering Internship, Week 5

    Security Agent      Performance Agent      Maintainability Agent
          \                    |                    /
           \                   |                   /
          (Independent Round 1 Review - No Inter-Agent Discussion)
                               |
                               v
                     Consensus Calculation
                       /              \
         Share >= 60% /                \ Share < 60%
                     /                  \
            [Round 1 Final]       Top 2 Specialists Only
                                        |
                                        v
                                 (Round 2 Re-Review)
                                        |
                                        v
                               [Round 2 Final (Max 2 Rounds)]

Why this consensus pattern exists:
Confidence-weighted consensus aggregates independent expert assessments without
allowing inter-agent debate beforehand. Allowing agents to debate before voting can
lead to peer pressure, premature convergence, or dominance by noisy agents. Keeping
specialists strictly independent until voting ensures genuine diversity of perspective.

How the consensus engine operates:
1. Three specialists (Security, Performance, Maintainability) review code independently.
2. Each returns a SpecialistOpinion (verdict: approve/needs_changes/reject, confidence 0..1, reasoning, findings).
3. The engine calculates the weighted confidence share for each verdict:
      Share(v) = (Sum of confidence of specialists choosing v) / (Sum of all specialist confidence)
4. If the leading verdict share >= 0.60, it becomes the final verdict in Round 1.
5. If < 0.60, the system escalates to Round 2 using ONLY the top 2 specialists with the
   highest individual confidence scores. The lowest-confidence specialist is excluded.
6. The top 2 specialists receive a neutral summary of Round 1 disagreement and restate their opinion.
7. Weighted share is recomputed from Round 2. If still < 0.60, the leading verdict is output with
   cleared_threshold = False. The engine NEVER loops past Round 2.

Usage:
    python consensus_engine.py sample.py
    python consensus_engine.py sample.py --real
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, TypedDict

import typer
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

app = typer.Typer(help="Lab 5.3: Multi-perspective code reviewer and confidence-weighted consensus engine.")
console = Console()


# ---------------------------------------------------------------------------
# Typed Data Models (Pydantic)
# ---------------------------------------------------------------------------


class SpecialistOpinion(BaseModel):
    """An independent review produced by one specialist role (Security, Performance, or Maintainability)."""

    specialist_name: str = Field(description="Name of the specialist role.")
    verdict: Literal["approve", "needs_changes", "reject"] = Field(
        description="The recommended code review verdict."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0."
    )
    reasoning: str = Field(description="Detailed technical reasoning for the verdict.")
    findings: list[str] = Field(
        default_factory=list, description="Specific findings or concerns identified."
    )


class ConsensusVerdict(BaseModel):
    """The aggregated consensus verdict produced by the Consensus Engine."""

    final_verdict: Literal["approve", "needs_changes", "reject"] = Field(
        description="The winning aggregated verdict."
    )
    weighted_confidence_share: float = Field(
        ge=0.0, le=1.0, description="The weighted confidence share (0.0 to 1.0) of the winning verdict."
    )
    rounds_used: Literal[1, 2] = Field(
        description="The number of review rounds executed (1 or 2)."
    )
    cleared_threshold: bool = Field(
        description="True if the winning verdict reached or exceeded the 60% (0.60) confidence threshold."
    )
    conflict_annotations: list[str] = Field(
        default_factory=list,
        description="Detailed annotations of verdict splits and unique specialist findings.",
    )
    dissent_summary: str = Field(
        description="Summary of dissenting views. Empty string if all 3 specialists agreed in Round 1."
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------


class ConsensusState(TypedDict):
    """LangGraph execution state tracking review rounds, opinions, and consensus verdict."""

    code_content: str
    round_1_opinions: list[SpecialistOpinion]
    round_2_opinions: list[SpecialistOpinion]
    rounds_used: int
    top_2_names: list[str]
    neutral_summary: str
    verdict: ConsensusVerdict | None


# ---------------------------------------------------------------------------
# Consensus Aggregation Engine Core Logic
# ---------------------------------------------------------------------------


def compute_weighted_share(opinions: list[SpecialistOpinion]) -> tuple[Literal["approve", "needs_changes", "reject"], float]:
    """Computes the weighted confidence share for each verdict and returns (winning_verdict, winning_share).

    Formula:
        Share(v) = (Sum of confidence of specialists choosing v) / (Sum of all active confidence)
    """
    if not opinions:
        return "approve", 0.0

    total_confidence = sum(op.confidence for op in opinions)
    if total_confidence <= 0.0:
        # Fallback to simple majority vote if total confidence is zero
        verdict_counts: dict[str, int] = {}
        for op in opinions:
            verdict_counts[op.verdict] = verdict_counts.get(op.verdict, 0) + 1
        best_v = max(verdict_counts, key=lambda k: verdict_counts[k])
        return best_v, 1.0 / len(opinions)

    verdict_weighted_sums: dict[Literal["approve", "needs_changes", "reject"], float] = {
        "approve": 0.0,
        "needs_changes": 0.0,
        "reject": 0.0,
    }

    for op in opinions:
        verdict_weighted_sums[op.verdict] += op.confidence

    # Find verdict with maximum weighted sum
    winning_verdict = max(verdict_weighted_sums, key=lambda v: verdict_weighted_sums[v])
    winning_share = verdict_weighted_sums[winning_verdict] / total_confidence

    return winning_verdict, winning_share


def build_conflict_annotations_and_dissent(
    round_1_opinions: list[SpecialistOpinion],
) -> tuple[list[str], str]:
    """Generates conflict annotations and dissent summary.

    Rule: dissent_summary is an empty string ("") ONLY IF all 3 specialists agreed on the exact same verdict in round 1.
    """
    verdicts = [op.verdict for op in round_1_opinions]
    all_agreed = len(set(verdicts)) == 1 if round_1_opinions else True

    conflict_annotations: list[str] = []
    for op in round_1_opinions:
        conflict_annotations.append(
            f"{op.specialist_name}: Voted '{op.verdict}' with confidence {op.confidence:.2f}. "
            f"Key reasoning: {op.reasoning[:80]}..."
        )
        for finding in op.findings:
            conflict_annotations.append(f"  - [{op.specialist_name} Finding] {finding}")

    if all_agreed:
        dissent_summary = ""
    else:
        dissent_parts = []
        for op in round_1_opinions:
            dissent_parts.append(
                f"{op.specialist_name} voted '{op.verdict}' (conf: {op.confidence:.2f}): {op.reasoning}"
            )
        dissent_summary = "Disagreement among specialists in Round 1: " + " | ".join(dissent_parts)

    return conflict_annotations, dissent_summary


# ---------------------------------------------------------------------------
# Stub Heuristics (Deterministic Input-Sensitive Reviewers)
# ---------------------------------------------------------------------------


def stub_security_reviewer(code_content: str) -> SpecialistOpinion:
    """Security specialist stub inspecting code for vulnerability indicators."""
    findings = []
    code_lower = code_content.lower()

    if "password" in code_lower or "secret" in code_lower or "api_key" in code_lower:
        findings.append("Potential hardcoded credential or secret detected.")
    if "select " in code_lower and (" + " in code_lower or " % " in code_lower or "f\"" in code_lower or "f'" in code_lower):
        findings.append("Potential SQL injection vulnerability via dynamic string formatting.")
    if "eval(" in code_lower or "exec(" in code_lower:
        findings.append("Unsafe dynamic code execution (eval/exec) detected.")

    if len(findings) >= 2:
        return SpecialistOpinion(
            specialist_name="Security Agent",
            verdict="reject",
            confidence=0.90,
            reasoning="Critical security flaws detected including hardcoded credentials or injection vectors.",
            findings=findings,
        )
    elif len(findings) == 1:
        return SpecialistOpinion(
            specialist_name="Security Agent",
            verdict="needs_changes",
            confidence=0.75,
            reasoning="Security concern identified requiring remediation before approval.",
            findings=findings,
        )
    else:
        return SpecialistOpinion(
            specialist_name="Security Agent",
            verdict="approve",
            confidence=0.85,
            reasoning="No obvious security vulnerabilities, hardcoded secrets, or SQL injection vectors found.",
            findings=["Security check passed cleanly."],
        )


def stub_performance_reviewer(code_content: str) -> SpecialistOpinion:
    """Performance specialist stub inspecting code for performance bottlenecks."""
    findings = []
    lines = code_content.splitlines()

    # Check for nested loops
    loop_depth = 0
    max_loop_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("for ") or stripped.startswith("while "):
            loop_depth += 1
            max_loop_depth = max(max_loop_depth, loop_depth)
        elif stripped and not stripped.startswith("#") and len(line) - len(line.lstrip()) == 0:
            loop_depth = 0

    if max_loop_depth >= 2:
        findings.append(f"Nested loop structure detected (depth {max_loop_depth}), potential O(N^2) latency impact.")
    if "open(" in code_content and ("for " in code_content or "while " in code_content):
        findings.append("File I/O inside loop construct detected, high disk/network latency risk.")
    if "requests." in code_content or "http" in code_content:
        findings.append("Network calls without explicit timeout parameter identified.")

    if len(findings) >= 2:
        return SpecialistOpinion(
            specialist_name="Performance Agent",
            verdict="needs_changes",
            confidence=0.80,
            reasoning="Multiple performance bottlenecks found (nested loops / repeated I/O).",
            findings=findings,
        )
    elif len(findings) == 1:
        return SpecialistOpinion(
            specialist_name="Performance Agent",
            verdict="needs_changes",
            confidence=0.70,
            reasoning="Performance degradation risk identified.",
            findings=findings,
        )
    else:
        return SpecialistOpinion(
            specialist_name="Performance Agent",
            verdict="approve",
            confidence=0.85,
            reasoning="Code avoids nested loop complexity and repeated synchronous I/O.",
            findings=["Performance review passed cleanly."],
        )


def stub_maintainability_reviewer(code_content: str) -> SpecialistOpinion:
    """Maintainability specialist stub inspecting code for readability and style."""
    findings = []
    lines = code_content.splitlines()

    if '"""' not in code_content and "'''" not in code_content:
        findings.append("Missing module/function docstrings.")
    if len(lines) > 40:
        findings.append(f"Large file size ({len(lines)} lines), consider modularizing.")
    if "def " in code_content and "->" not in code_content:
        findings.append("Missing type hint annotations on function signatures.")

    if len(findings) >= 2:
        return SpecialistOpinion(
            specialist_name="Maintainability Agent",
            verdict="needs_changes",
            confidence=0.75,
            reasoning="Code documentation and type annotation standards are insufficient.",
            findings=findings,
        )
    elif len(findings) == 1:
        return SpecialistOpinion(
            specialist_name="Maintainability Agent",
            verdict="needs_changes",
            confidence=0.65,
            reasoning="Minor maintainability improvements recommended.",
            findings=findings,
        )
    else:
        return SpecialistOpinion(
            specialist_name="Maintainability Agent",
            verdict="approve",
            confidence=0.85,
            reasoning="Code includes proper docstrings, type annotations, and clean structure.",
            findings=["Maintainability review passed cleanly."],
        )


# ---------------------------------------------------------------------------
# Real Groq Reviewers (--real mode)
# ---------------------------------------------------------------------------


def _get_groq_client():
    """Returns a ChatGroq instance or raises RuntimeError if API key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is missing. "
            "Please set GROQ_API_KEY in your .env file or environment to run with --real."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def real_specialist_review(
    specialist_name: str,
    system_prompt: str,
    code_content: str,
    neutral_summary: str | None = None,
) -> SpecialistOpinion:
    """Calls Groq LLM with structured output for a specialist review."""
    llm = _get_groq_client()
    structured_llm = llm.with_structured_output(SpecialistOpinion)

    user_prompt = f"Code to review:\n```python\n{code_content}\n```\n"
    if neutral_summary:
        user_prompt += f"\nRound 1 Disagreement Summary (for Round 2 Re-evaluation):\n{neutral_summary}\n"
    user_prompt += f"\nProvide your SpecialistOpinion as the {specialist_name}."

    res = structured_llm.invoke([("system", system_prompt), ("human", user_prompt)])
    if not isinstance(res, SpecialistOpinion):
        res = SpecialistOpinion.model_validate(res)
    res.specialist_name = specialist_name
    return res


# ---------------------------------------------------------------------------
# LangGraph Workflow Nodes & Graph Construction
# ---------------------------------------------------------------------------


def make_round_1_node(
    is_real: bool = False,
    security_provider: Callable[[str], SpecialistOpinion] | None = None,
    performance_provider: Callable[[str], SpecialistOpinion] | None = None,
    maintainability_provider: Callable[[str], SpecialistOpinion] | None = None,
) -> Callable[[ConsensusState], dict]:
    """Node that executes independent Round 1 reviews for all 3 specialists."""

    def round_1_node(state: ConsensusState) -> dict:
        code = state["code_content"]

        # 1. Security review
        if security_provider:
            sec_op = security_provider(code)
        elif not is_real:
            sec_op = stub_security_reviewer(code)
        else:
            sec_prompt = (
                "You are an expert Security Agent reviewing code strictly for vulnerabilities, "
                "hardcoded secrets, SQL injection, unsafe deserialization, and access control risks."
            )
            sec_op = real_specialist_review("Security Agent", sec_prompt, code)

        # 2. Performance review
        if performance_provider:
            perf_op = performance_provider(code)
        elif not is_real:
            perf_op = stub_performance_reviewer(code)
        else:
            perf_prompt = (
                "You are an expert Performance Agent reviewing code strictly for algorithmic efficiency, "
                "nested loops, IO bottlenecks, memory usage, and execution latency."
            )
            perf_op = real_specialist_review("Performance Agent", perf_prompt, code)

        # 3. Maintainability review
        if maintainability_provider:
            maint_op = maintainability_provider(code)
        elif not is_real:
            maint_op = stub_maintainability_reviewer(code)
        else:
            maint_prompt = (
                "You are an expert Maintainability Agent reviewing code strictly for code readability, "
                "docstrings, clean functions, modularity, and type annotations."
            )
            maint_op = real_specialist_review("Maintainability Agent", maint_prompt, code)

        opinions = [sec_op, perf_op, maint_op]
        return {"round_1_opinions": opinions}

    return round_1_node


def evaluate_round_1_consensus(state: ConsensusState) -> dict:
    """Evaluates Round 1 weighted confidence share and prepares escalation if < 0.60."""
    opinions = state["round_1_opinions"]
    winning_verdict, winning_share = compute_weighted_share(opinions)
    conflict_annotations, dissent_summary = build_conflict_annotations_and_dissent(opinions)

    if winning_share >= 0.60:
        verdict = ConsensusVerdict(
            final_verdict=winning_verdict,
            weighted_confidence_share=winning_share,
            rounds_used=1,
            cleared_threshold=True,
            conflict_annotations=conflict_annotations,
            dissent_summary=dissent_summary,
        )
        return {"rounds_used": 1, "verdict": verdict}

    # Sort specialists by confidence descending to pick top 2
    sorted_opinions = sorted(opinions, key=lambda op: op.confidence, reverse=True)
    top_2_ops = sorted_opinions[:2]
    top_2_names = [op.specialist_name for op in top_2_ops]

    # Neutral summary of Round 1 disagreement for Round 2
    disagreement_parts = [
        f"{op.specialist_name} voted '{op.verdict}' (confidence: {op.confidence:.2f})"
        for op in opinions
    ]
    neutral_summary = (
        f"Round 1 weighted confidence ({winning_share:.2%}) was below the 60.0% threshold. "
        f"Round 1 disagreement summary: {', '.join(disagreement_parts)}. "
        "Please re-evaluate your assessment in light of these perspective splits."
    )

    return {
        "rounds_used": 1,
        "top_2_names": top_2_names,
        "neutral_summary": neutral_summary,
    }


def route_after_round_1(state: ConsensusState) -> Literal["finalize_round_1", "escalate_to_round_2"]:
    """Conditional router checking if Round 1 cleared 0.60 threshold."""
    if state.get("verdict") is not None:
        return "finalize_round_1"
    return "escalate_to_round_2"


def make_round_2_node(
    is_real: bool = False,
    round_2_override_provider: Callable[[str, list[str], str], list[SpecialistOpinion]] | None = None,
) -> Callable[[ConsensusState], dict]:
    """Node that executes Round 2 for ONLY the top 2 specialists by Round 1 confidence."""

    def round_2_node(state: ConsensusState) -> dict:
        code = state["code_content"]
        top_2_names = state["top_2_names"]
        neutral_summary = state["neutral_summary"]
        r1_ops = {op.specialist_name: op for op in state["round_1_opinions"]}

        if round_2_override_provider:
            r2_opinions = round_2_override_provider(code, top_2_names, neutral_summary)
            return {"round_2_opinions": r2_opinions, "rounds_used": 2}

        r2_opinions: list[SpecialistOpinion] = []
        for name in top_2_names:
            r1_op = r1_ops[name]
            if not is_real:
                # Stub round 2: reconsider slightly adjusted confidence or restate verdict
                # For deterministic testing, stub returns restated/refined opinion based on neutral summary
                r2_op = SpecialistOpinion(
                    specialist_name=name,
                    verdict=r1_op.verdict,
                    confidence=min(1.0, r1_op.confidence + 0.05),
                    reasoning=f"[Round 2 Re-evaluation] Re-affirming {r1_op.verdict} after considering: {neutral_summary[:60]}...",
                    findings=r1_op.findings,
                )
            else:
                prompt_map = {
                    "Security Agent": "You are the Security Agent re-evaluating code in Round 2.",
                    "Performance Agent": "You are the Performance Agent re-evaluating code in Round 2.",
                    "Maintainability Agent": "You are the Maintainability Agent re-evaluating code in Round 2.",
                }
                r2_op = real_specialist_review(name, prompt_map[name], code, neutral_summary)

            r2_opinions.append(r2_op)

        return {"round_2_opinions": r2_opinions, "rounds_used": 2}

    return round_2_node


def evaluate_round_2_consensus(state: ConsensusState) -> dict:
    """Evaluates Round 2 consensus share. Hard cap at 2 rounds maximum."""
    r1_opinions = state["round_1_opinions"]
    r2_opinions = state["round_2_opinions"]

    winning_verdict, winning_share = compute_weighted_share(r2_opinions)
    conflict_annotations, dissent_summary = build_conflict_annotations_and_dissent(r1_opinions)

    cleared = winning_share >= 0.60
    verdict = ConsensusVerdict(
        final_verdict=winning_verdict,
        weighted_confidence_share=winning_share,
        rounds_used=2,
        cleared_threshold=cleared,
        conflict_annotations=conflict_annotations,
        dissent_summary=dissent_summary,
    )
    return {"verdict": verdict}


def build_consensus_graph(
    is_real: bool = False,
    security_provider: Callable[[str], SpecialistOpinion] | None = None,
    performance_provider: Callable[[str], SpecialistOpinion] | None = None,
    maintainability_provider: Callable[[str], SpecialistOpinion] | None = None,
    round_2_override_provider: Callable[[str, list[str], str], list[SpecialistOpinion]] | None = None,
) -> StateGraph:
    """Builds and compiles the Consensus Engine StateGraph."""
    builder = StateGraph(ConsensusState)

    r1_node = make_round_1_node(
        is_real=is_real,
        security_provider=security_provider,
        performance_provider=performance_provider,
        maintainability_provider=maintainability_provider,
    )
    r2_node = make_round_2_node(
        is_real=is_real,
        round_2_override_provider=round_2_override_provider,
    )

    builder.add_node("round_1_reviewers", r1_node)
    builder.add_node("eval_round_1", evaluate_round_1_consensus)
    builder.add_node("round_2_reviewers", r2_node)
    builder.add_node("eval_round_2", evaluate_round_2_consensus)

    builder.add_edge(START, "round_1_reviewers")
    builder.add_edge("round_1_reviewers", "eval_round_1")

    builder.add_conditional_edges(
        "eval_round_1",
        route_after_round_1,
        {
            "finalize_round_1": END,
            "escalate_to_round_2": "round_2_reviewers",
        },
    )

    builder.add_edge("round_2_reviewers", "eval_round_2")
    builder.add_edge("eval_round_2", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main Execution Runner
# ---------------------------------------------------------------------------


def run_consensus(
    code_content: str,
    is_real: bool = False,
    security_provider: Callable[[str], SpecialistOpinion] | None = None,
    performance_provider: Callable[[str], SpecialistOpinion] | None = None,
    maintainability_provider: Callable[[str], SpecialistOpinion] | None = None,
    round_2_override_provider: Callable[[str, list[str], str], list[SpecialistOpinion]] | None = None,
) -> tuple[list[SpecialistOpinion], ConsensusVerdict]:
    """Executes the consensus engine graph on code content and returns (round_1_opinions, verdict)."""
    graph = build_consensus_graph(
        is_real=is_real,
        security_provider=security_provider,
        performance_provider=performance_provider,
        maintainability_provider=maintainability_provider,
        round_2_override_provider=round_2_override_provider,
    )

    initial_state: ConsensusState = {
        "code_content": code_content,
        "round_1_opinions": [],
        "round_2_opinions": [],
        "rounds_used": 0,
        "top_2_names": [],
        "neutral_summary": "",
        "verdict": None,
    }

    final_state = graph.invoke(initial_state)
    verdict = final_state["verdict"]
    assert verdict is not None, "ConsensusVerdict must not be None"
    return final_state["round_1_opinions"], verdict


# ---------------------------------------------------------------------------
# Typer CLI Entry Point
# ---------------------------------------------------------------------------


@app.command()
def main(
    code_path: str = typer.Argument(
        ..., help="Path to the source code file to review."
    ),
    real: bool = typer.Option(
        False, "--real", help="Use real Groq LLM calls (llama-3.3-70b-versatile) instead of offline stubs."
    ),
):
    """Run multi-perspective code review and confidence-weighted consensus engine."""
    file_p = Path(code_path)
    if not file_p.exists():
        console.print(f"[bold red]Error:[/bold red] File '{code_path}' does not exist.")
        raise typer.Exit(code=1)

    code_content = file_p.read_text(encoding="utf-8")
    mode_str = "Real Groq LLM (llama-3.3-70b-versatile)" if real else "Offline Stub (Rule Heuristics)"

    console.print(
        Panel.fit(
            f"[bold cyan]Lab 5.3: Confidence-Weighted Consensus Engine[/bold cyan]\n"
            f"[bold yellow]Code File:[/bold yellow] {code_path}\n"
            f"[bold green]Mode:[/bold green] {mode_str}",
            border_style="blue",
        )
    )

    r1_opinions, verdict = run_consensus(code_content, is_real=real)

    # Render Side-by-Side Specialist Opinions Table
    table = Table(title="Specialist Independent Reviews (Round 1)", border_style="cyan")
    table.add_column("Specialist", style="bold white", width=22)
    table.add_column("Verdict", style="bold", width=16)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Top Finding / Reasoning Summary", style="dim", width=45)

    for op in r1_opinions:
        verdict_color = "green" if op.verdict == "approve" else ("yellow" if op.verdict == "needs_changes" else "red")
        top_finding = op.findings[0] if op.findings else op.reasoning[:40]
        table.add_row(
            op.specialist_name,
            f"[{verdict_color}]{op.verdict.upper()}[/{verdict_color}]",
            f"{op.confidence:.2f}",
            top_finding,
        )

    console.print(table)

    # Render Consensus Verdict Panel
    final_color = "green" if verdict.final_verdict == "approve" else ("yellow" if verdict.final_verdict == "needs_changes" else "red")
    threshold_str = "[bold green]YES[/bold green]" if verdict.cleared_threshold else "[bold red]NO (Unresolved Split)[/bold red]"

    annotations_str = "\n".join(verdict.conflict_annotations)
    dissent_display = verdict.dissent_summary if verdict.dissent_summary else "[italic gray]None (Unanimous Round 1 Agreement)[/italic gray]"

    verdict_content = (
        f"[bold yellow]Final Consensus Verdict:[/bold yellow] [{final_color}]{verdict.final_verdict.upper()}[/{final_color}]\n"
        f"[bold yellow]Weighted Confidence Share:[/bold yellow] {verdict.weighted_confidence_share:.2%}\n"
        f"[bold yellow]Rounds Executed:[/bold yellow] {verdict.rounds_used} of 2 (Max)\n"
        f"[bold yellow]Cleared >=60% Threshold:[/bold yellow] {threshold_str}\n\n"
        f"[bold yellow]Dissent Summary:[/bold yellow]\n{dissent_display}\n\n"
        f"[bold yellow]Conflict Annotations:[/bold yellow]\n{annotations_str}"
    )

    console.print(
        Panel(
            verdict_content,
            title="[bold green]FINAL CONSENSUS VERDICT[/bold green]",
            border_style=final_color,
        )
    )


if __name__ == "__main__":
    app()
