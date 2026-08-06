"""
Week 5, Day 4 (Thursday Core Learning) -- Debate Trio: Proposer, Challenger, Arbiter
CalderR Agentic AI Engineering Internship, Week 5

    Round 0: Proposer states initial position with supporting reasoning.
    Round 1: Challenger reads Proposer's position and finds the weakest point.
    Round 2: Proposer responds directly to Challenger's specific point.
    Round 3: Challenger makes a final rebuttal point.
    Post-Round 3: Arbiter evaluates the full transcript and renders a verdict.

Why this debate pattern exists:
Multi-agent debate is designed to adversarially pressure-test a single claim
or architecture decision, contrasting with multi-agent consensus (which blends
multiple independent expert opinions).

A major documented vulnerability in LLM debate judges is RECENCY BIAS: LLM judges
evaluating multi-turn transcripts tend to fixate on whichever message they read
most recently, under-weighting earlier substantive arguments. An Arbiter that lacks
explicit resistance to recency bias defaults to picking whoever spoke last.

This lab builds a 3-agent debate graph where:
1. Proposer and Challenger execute a fixed 4-statement sequence (rounds 0..3).
2. The Arbiter evaluates argument quality independently of position or recency.
3. In offline stub mode, a content-based rubric scores each statement's substance
   (specificity, concrete evidence indicators, logical depth) rather than position.
4. In real LLM mode (--real), system prompts explicitly instruct the Arbiter that
   recency is not evidence of correctness.

Usage:
    python debate_trio.py "Should microservices be preferred over modular monoliths for early-stage startups?"
    python debate_trio.py "Should Python use static typing by default?" --real
"""

from __future__ import annotations

import os
from typing import Callable, Literal, TypedDict

import typer
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

load_dotenv()

app = typer.Typer(help="Debate Trio CLI: Proposer, Challenger, and anti-recency Arbiter.")
console = Console()


# ---------------------------------------------------------------------------
# Typed Data Models (Pydantic)
# ---------------------------------------------------------------------------


class Statement(BaseModel):
    """A single argument turn made by either the Proposer or Challenger."""

    round: int = Field(description="The zero-indexed round number (0..3).")
    role: Literal["proposer", "challenger"] = Field(description="The role of the debater.")
    content: str = Field(description="The textual content of the argument.")


class DebateTranscript(BaseModel):
    """The full recorded transcript of all debate rounds for a given question."""

    question: str = Field(description="The core claim or question under debate.")
    statements: list[Statement] = Field(
        default_factory=list, description="Ordered list of statements across rounds 0..3."
    )


class ArbiterVerdict(BaseModel):
    """The final impartial judgment rendered by the Arbiter."""

    winning_side: Literal["proposer", "challenger"] = Field(
        description="The debater side that presented the stronger overall case."
    )
    decisive_round: int = Field(
        description="The specific round (0..3) containing the decisive argument."
    )
    reasoning: str = Field(
        description="Detailed explanation of why the winning side's argument prevailed based on logical quality, not recency."
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------


class DebateState(TypedDict):
    """LangGraph execution state tracking the question, accumulated statements, and verdict."""

    question: str
    statements: list[Statement]
    verdict: ArbiterVerdict | None


# ---------------------------------------------------------------------------
# Content-Based Rubric (Anti-Recency Bias Engine for Offline Stub)
# ---------------------------------------------------------------------------


def score_statement_content(content: str) -> float:
    """Evaluates the substantive quality of an argument string purely from content.

    Why this function exists:
    A stub Arbiter that evaluates statements based on index (e.g. returning statements[-1])
    reintroduces the exact recency-bias defect this lab is designed to eliminate.
    This rubric measures objective substance: length/depth, specific evidence markers,
    concrete technical detail indicators, and structural rigor while penalizing generic
    or vague filler.
    """
    text = content.strip()
    if not text:
        return 0.0

    score = 0.0
    words = text.split()
    word_count = len(words)

    # 1. Depth & elaboration (up to 3.0 pts)
    score += min(3.0, word_count / 15.0)

    # 2. Specific evidence & concrete domain markers (0.8 pts each)
    substantive_indicators = [
        "benchmark",
        "latency",
        "throughput",
        "tradeoff",
        "concrete",
        "evidence",
        "metric",
        "failure mode",
        "operational overhead",
        "coupling",
        "schema",
        "type safety",
        "refactoring",
        "profiling",
        "bottleneck",
        "invariant",
    ]
    text_lower = text.lower()
    for keyword in substantive_indicators:
        if keyword in text_lower:
            score += 0.8

    # 3. Direct rebuttal / engagement markers (0.6 pts each)
    rebuttal_indicators = [
        "specifically",
        "however",
        "contrary to",
        "fails to address",
        "disprove",
        "concede",
        "demonstrates",
    ]
    for keyword in rebuttal_indicators:
        if keyword in text_lower:
            score += 0.6

    # 4. Penalty for generic, low-substance filler (-1.5 pts each)
    generic_phrases = [
        "i just disagree",
        "whatever",
        "my position is obvious",
        "generic statement",
        "no comment",
        "as i said before",
        "simply wrong without explanation",
    ]
    for phrase in generic_phrases:
        if phrase in text_lower:
            score -= 1.5

    return max(0.0, score)


def stub_arbiter_eval(transcript: DebateTranscript) -> ArbiterVerdict:
    """Determines the debate winner using the content-based rubric across all statements.

    Position independence guarantee:
    Every statement in the transcript is scored solely by `score_statement_content(content)`.
    The winner is the role associated with the highest-scoring single statement.
    Statement index / recency is NEVER used to select the winner.
    """
    if not transcript.statements:
        return ArbiterVerdict(
            winning_side="proposer",
            decisive_round=0,
            reasoning="No statements recorded in transcript.",
        )

    # Score each statement strictly by content
    scored_statements = [
        (stmt, score_statement_content(stmt.content)) for stmt in transcript.statements
    ]

    # Find statement with highest substantive content score
    best_stmt, best_score = max(scored_statements, key=lambda x: x[1])

    role = best_stmt.role
    decisive_round = best_stmt.round

    reasoning = (
        f"Round {decisive_round} ({role.capitalize()}) delivered the highest-quality substantive argument "
        f"(content score: {best_score:.2f}). The evaluation prioritizes logical depth and specific evidence "
        f"over statement recency."
    )

    return ArbiterVerdict(
        winning_side=role,
        decisive_round=decisive_round,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Default Fixture Generators for Stub Mode
# ---------------------------------------------------------------------------


def default_stub_proposer_response(question: str, round_num: int, previous_statements: list[Statement]) -> str:
    """Generates deterministic Proposer arguments for stub mode."""
    if round_num == 0:
        return (
            f"Proposer initial position on '{question}': We strongly advocate for this approach because "
            "it establishes clear type safety, reduces runtime failure modes, and improves developer velocity "
            "through concrete compile-time contracts and automated refactoring benchmarks."
        )
    elif round_num == 2:
        prev_challenger = previous_statements[-1].content if previous_statements else ""
        return (
            f"Proposer round 2 defense: Responding specifically to the challenger's concern regarding '{prev_challenger[:40]}...': "
            "While operational overhead exists during initial setup, the reduction in production latency bottlenecks, "
            "schema migration errors, and coupling metrics far outweighs the initial investment."
        )
    return f"Proposer argument for round {round_num} on {question}."


def default_stub_challenger_response(question: str, round_num: int, previous_statements: list[Statement]) -> str:
    """Generates deterministic Challenger arguments for stub mode."""
    prev_proposer = previous_statements[-1].content if previous_statements else ""
    if round_num == 1:
        return (
            f"Challenger round 1 counter-argument: Examining the proposer's claim '{prev_proposer[:40]}...': "
            "The proposer overstates the benefits. In early-stage development, rigid schema constraints introduce "
            "significant operational overhead, increase friction during rapid iteration, and fail to address "
            "dynamic payload flexibility."
        )
    elif round_num == 3:
        return (
            "Challenger round 3 final point: In conclusion, flexibility and iteration velocity trump premature formalization. "
            "Generic statement without new benchmark data or specific operational evidence."
        )
    return f"Challenger argument for round {round_num} on {question}."


# ---------------------------------------------------------------------------
# Agent Node Factories (LangGraph Nodes)
# ---------------------------------------------------------------------------


def make_proposer_node(
    is_real: bool = False,
    custom_provider: Callable[[str, int, list[Statement]], str] | None = None,
) -> Callable[[DebateState], dict]:
    """Creates the Proposer node function."""

    def proposer_node(state: DebateState) -> dict:
        question = state["question"]
        current_statements = list(state.get("statements", []))
        round_num = len(current_statements)

        if custom_provider:
            content = custom_provider(question, round_num, current_statements)
        elif not is_real:
            content = default_stub_proposer_response(question, round_num, current_statements)
        else:
            content = _call_groq_proposer(question, round_num, current_statements)

        stmt = Statement(round=round_num, role="proposer", content=content)
        return {"statements": current_statements + [stmt]}

    return proposer_node


def make_challenger_node(
    is_real: bool = False,
    custom_provider: Callable[[str, int, list[Statement]], str] | None = None,
) -> Callable[[DebateState], dict]:
    """Creates the Challenger node function."""

    def challenger_node(state: DebateState) -> dict:
        question = state["question"]
        current_statements = list(state.get("statements", []))
        round_num = len(current_statements)

        if custom_provider:
            content = custom_provider(question, round_num, current_statements)
        elif not is_real:
            content = default_stub_challenger_response(question, round_num, current_statements)
        else:
            content = _call_groq_challenger(question, round_num, current_statements)

        stmt = Statement(round=round_num, role="challenger", content=content)
        return {"statements": current_statements + [stmt]}

    return challenger_node


def make_arbiter_node(
    is_real: bool = False,
) -> Callable[[DebateState], dict]:
    """Creates the Arbiter node function."""

    def arbiter_node(state: DebateState) -> dict:
        question = state["question"]
        statements = state["statements"]
        transcript = DebateTranscript(question=question, statements=statements)

        if not is_real:
            verdict = stub_arbiter_eval(transcript)
        else:
            verdict = _call_groq_arbiter(transcript)

        return {"verdict": verdict}

    return arbiter_node


# ---------------------------------------------------------------------------
# Real Groq LLM Calls (--real mode)
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


def _call_groq_proposer(question: str, round_num: int, previous_statements: list[Statement]) -> str:
    llm = _get_groq_client()
    system_prompt = (
        "You are an expert technical Proposer participating in a structured debater trio.\n"
        "Your goal is to present and defend the position as strongly as legitimately possible.\n"
        "Rules:\n"
        "- In Round 0: Present a clear initial position backed by concrete reasoning, trade-off analysis, and metrics.\n"
        "- In Round 2: Read the Challenger's point carefully. Respond directly to the specific weak point raised — defend, concede, or refine. Do NOT just repeat your opening statement."
    )

    history_str = "\n".join(
        f"Round {s.round} [{s.role.upper()}]: {s.content}" for s in previous_statements
    )
    user_prompt = f"Question/Claim: {question}\n\nTranscript so far:\n{history_str}\n\nDeliver your argument for Round {round_num}:"

    response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
    return str(response.content).strip()


def _call_groq_challenger(question: str, round_num: int, previous_statements: list[Statement]) -> str:
    llm = _get_groq_client()
    system_prompt = (
        "You are an expert technical Challenger in a structured debater trio.\n"
        "Your explicit job is to find and press the actual weakest point in the Proposer's case.\n"
        "Rules:\n"
        "- Do NOT agree or restate the Proposer's position.\n"
        "- In Round 1: Target the single weakest assumption or failure mode in the Proposer's Round 0 opening.\n"
        "- In Round 3: Make your final rebuttal point pressing why your challenge holds."
    )

    history_str = "\n".join(
        f"Round {s.round} [{s.role.upper()}]: {s.content}" for s in previous_statements
    )
    user_prompt = f"Question/Claim: {question}\n\nTranscript so far:\n{history_str}\n\nDeliver your counter-argument for Round {round_num}:"

    response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
    return str(response.content).strip()


def _call_groq_arbiter(transcript: DebateTranscript) -> ArbiterVerdict:
    llm = _get_groq_client()
    structured_llm = llm.with_structured_output(ArbiterVerdict)

    system_prompt = (
        "You are an impartial Arbiter evaluating a 4-statement technical debate transcript.\n"
        "Your task is to analyze the argument quality of each statement independently.\n\n"
        "CRITICAL INSTRUCTION ON RECENCY BIAS:\n"
        "Recency is NOT evidence of correctness. Do NOT give extra weight to whichever side spoke last.\n"
        "Evaluate each statement strictly on logical soundness, concrete evidence, and directness of rebuttal.\n"
        "Determine which side presented the stronger overall case, identify the exact decisive round (0, 1, 2, or 3), "
        "and explain your reasoning."
    )

    history_str = "\n".join(
        f"Round {s.round} [{s.role.upper()}]: {s.content}" for s in transcript.statements
    )
    user_prompt = (
        f"Debate Question: {transcript.question}\n\n"
        f"Full Transcript (Rounds 0..3):\n{history_str}\n\n"
        "Render your ArbiterVerdict."
    )

    verdict = structured_llm.invoke([("system", system_prompt), ("human", user_prompt)])
    if not isinstance(verdict, ArbiterVerdict):
        # Fallback dictionary coercion if structured output returned dict
        verdict = ArbiterVerdict.model_validate(verdict)
    return verdict


# ---------------------------------------------------------------------------
# LangGraph Workflow Construction
# ---------------------------------------------------------------------------


def build_debate_graph(
    is_real: bool = False,
    proposer_provider: Callable[[str, int, list[Statement]], str] | None = None,
    challenger_provider: Callable[[str, int, list[Statement]], str] | None = None,
) -> StateGraph:
    """Builds and compiles the 4-round debate StateGraph.

    Sequence: START -> proposer_0 -> challenger_1 -> proposer_2 -> challenger_3 -> arbiter -> END
    """
    builder = StateGraph(DebateState)

    p_node = make_proposer_node(is_real=is_real, custom_provider=proposer_provider)
    c_node = make_challenger_node(is_real=is_real, custom_provider=challenger_provider)
    a_node = make_arbiter_node(is_real=is_real)

    builder.add_node("proposer_0", p_node)
    builder.add_node("challenger_1", c_node)
    builder.add_node("proposer_2", p_node)
    builder.add_node("challenger_3", c_node)
    builder.add_node("arbiter", a_node)

    builder.add_edge(START, "proposer_0")
    builder.add_edge("proposer_0", "challenger_1")
    builder.add_edge("challenger_1", "proposer_2")
    builder.add_edge("proposer_2", "challenger_3")
    builder.add_edge("challenger_3", "arbiter")
    builder.add_edge("arbiter", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main Execution Runner
# ---------------------------------------------------------------------------


def run_debate(
    question: str,
    is_real: bool = False,
    proposer_provider: Callable[[str, int, list[Statement]], str] | None = None,
    challenger_provider: Callable[[str, int, list[Statement]], str] | None = None,
) -> tuple[DebateTranscript, ArbiterVerdict]:
    """Executes the debate graph and returns the complete transcript and arbiter verdict."""
    graph = build_debate_graph(
        is_real=is_real,
        proposer_provider=proposer_provider,
        challenger_provider=challenger_provider,
    )
    initial_state: DebateState = {
        "question": question,
        "statements": [],
        "verdict": None,
    }

    final_state = graph.invoke(initial_state)
    transcript = DebateTranscript(
        question=question, statements=final_state["statements"]
    )
    verdict = final_state["verdict"]
    assert verdict is not None, "Arbiter verdict must not be None"
    return transcript, verdict


# ---------------------------------------------------------------------------
# Typer CLI Entry Point
# ---------------------------------------------------------------------------


@app.command()
def main(
    question: str = typer.Argument(
        ..., help="The claim or technical question for the 3-agent debate."
    ),
    real: bool = typer.Option(
        False, "--real", help="Use real Groq LLM calls (llama-3.3-70b-versatile) instead of offline stub."
    ),
):
    """Run a 3-agent debate (Proposer, Challenger, Arbiter) with anti-recency bias protection."""
    mode_str = "Real Groq LLM (llama-3.3-70b-versatile)" if real else "Offline Stub (Deterministic Content Rubric)"
    console.print(
        Panel.fit(
            f"[bold cyan]Lab 5-5: Debate Trio[/bold cyan]\n"
            f"[bold yellow]Question:[/bold yellow] {question}\n"
            f"[bold green]Mode:[/bold green] {mode_str}",
            border_style="blue",
        )
    )

    transcript, verdict = run_debate(question, is_real=real)

    # Render each statement with distinct panel styling
    for stmt in transcript.statements:
        color = "cyan" if stmt.role == "proposer" else "magenta"
        title = f"Round {stmt.round} -- {stmt.role.upper()}"
        console.print(
            Panel(
                stmt.content,
                title=f"[bold {color}]{title}[/bold {color}]",
                border_style=color,
            )
        )

    # Render Arbiter Verdict panel
    winning_color = "cyan" if verdict.winning_side == "proposer" else "magenta"
    verdict_content = (
        f"[bold yellow]Winning Side:[/bold yellow] [{winning_color}]{verdict.winning_side.upper()}[/{winning_color}]\n"
        f"[bold yellow]Decisive Round:[/bold yellow] Round {verdict.decisive_round}\n\n"
        f"[bold yellow]Arbiter Reasoning:[/bold yellow]\n{verdict.reasoning}"
    )
    console.print(
        Panel(
            verdict_content,
            title="[bold green]ARBITER VERDICT (Anti-Recency Evaluation)[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
