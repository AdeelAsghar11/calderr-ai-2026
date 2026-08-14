r"""
Week 5, Day 2 (Tuesday) -- Supervisor Agent with Failure Recovery & Dynamic Rerouting
CalderR Agentic AI Engineering Internship, Week 5

    +-------------------+
    |     START         |
    +---------+---------+
              |
              v
    +-------------------+
    |     decompose     |
    +---------+---------+
              |
              v
+-->+-------------------+
|   |      attempt      |
|   +---------+---------+
|             |
|  [ conditional edge ]
|   /         |         \
|  / (pool    | (subtask \ (all subtasks
| / exhausted)| succeeded)\  resolved)
+-(retry next v            v
  specialist)+-----------+ +---------------+
             | (degrade &| |   aggregate   |
             |  advance) | +-------+-------+
             +-----+-----+         |
                   |               v
                   +------------> END

What makes this a *supervisor* rather than Lab 5.1's fixed pipeline:
In a fixed pipeline (such as research_agent -> synthesis_agent in lab-5-1), edge transitions
are static and predetermined at graph-construction time. In this supervisor architecture,
the graph contains dynamic routing logic (a conditional edge off `attempt`) where nodes inspect
the current runtime state—evaluating specialist outcomes, tracking tried specialists, failure
modes, and attempt counts—and dynamically decide whether to retry with another specialist,
degrade the subtask state, advance to the next subtask, or complete the workflow.

Usage:
    python supervisor_failure_recovery.py "Design and implement a secure user authentication microservice"
    python supervisor_failure_recovery.py "Build a resilient microservice" --seed 42
    python supervisor_failure_recovery.py "Build a resilient microservice" --real

(Typer collapses to a single positional argument -- TASK -- when only one command is registered,
so there's no subcommand word to type here; --real and --seed are options.)

--real switches specialist content generation to Groq (llama-3.3-70b-versatile via langchain-groq).
Requires GROQ_API_KEY in the environment. Default is the offline stub with simulated probabilistic
failures, running end-to-end with zero network access required -- see smoke_test.py.
"""

from __future__ import annotations

import operator
import os
import random
from typing import Annotated, Any, Callable, Literal, TypedDict

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langgraph.graph import END, START, StateGraph
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table

load_dotenv()

app = typer.Typer()
console = Console()

ALL_SPECIALISTS = ["Specialist A", "Specialist B", "Specialist C"]
CONFIDENCE_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Typed inter-agent models
# ---------------------------------------------------------------------------
class SpecialistResult(BaseModel):
    """Result returned by a specialist agent for a single subtask."""

    subtask: str
    specialist_name: str
    content: str
    confidence: float
    succeeded: bool


class DelegationDecision(BaseModel):
    """Structured record of a single delegation attempt or resolution by the supervisor."""

    subtask: str
    specialist_name: str
    attempt_number: int
    outcome: Literal["success", "timeout", "low_confidence", "exhausted"]
    reasoning: str


class SupervisorReport(BaseModel):
    """Final aggregated report produced by the supervisor for the entire task."""

    original_task: str
    subtask_results: list[SpecialistResult]
    degraded_subtasks: list[str]
    delegation_log: list[DelegationDecision]
    overall_status: Literal["complete", "degraded"]


# ---------------------------------------------------------------------------
# Specialist Runner with Probabilistic Failure Injection
# ---------------------------------------------------------------------------
class SpecialistRunner:
    """Executes specialist agents with configurable probabilistic failures for testing resilience."""

    def __init__(
        self,
        rng: random.Random | None = None,
        b_timeout_prob: float = 0.5,
        c_low_conf_prob: float = 0.5,
        use_real: bool = False,
        groq_client: Any | None = None,
    ):
        self.rng = rng or random.Random()
        self.b_timeout_prob = b_timeout_prob
        self.c_low_conf_prob = c_low_conf_prob
        self.use_real = use_real
        self.groq_client = groq_client

    def run_specialist(self, specialist_name: str, subtask: str) -> SpecialistResult:
        # Specialist B: Probabilistic timeout failure
        if specialist_name == "Specialist B":
            if self.rng.random() < self.b_timeout_prob:
                raise TimeoutError(f"Specialist B execution timed out after 5000ms for subtask: '{subtask}'")

        # Generate content
        if self.use_real and self.groq_client:
            try:
                prompt = f"You are {specialist_name}. Provide a 2-sentence technical response completing this subtask: {subtask}"
                response = self.groq_client.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
            except Exception:
                content = f"[{specialist_name} stub output for subtask: '{subtask}']"
        else:
            content = f"[{specialist_name} executed subtask successfully: '{subtask}']"

        # Determine confidence
        if specialist_name == "Specialist C":
            if self.rng.random() < self.c_low_conf_prob:
                confidence = round(self.rng.uniform(0.25, 0.55), 2)
            else:
                confidence = round(self.rng.uniform(0.70, 0.95), 2)
        elif specialist_name == "Specialist A":
            confidence = round(self.rng.uniform(0.85, 0.98), 2)
        else:  # Specialist B (when not timing out)
            confidence = round(self.rng.uniform(0.75, 0.92), 2)

        succeeded = confidence >= CONFIDENCE_THRESHOLD
        return SpecialistResult(
            subtask=subtask,
            specialist_name=specialist_name,
            content=content,
            confidence=confidence,
            succeeded=succeeded,
        )


def make_stub_runner(
    rng: random.Random | None = None,
    b_timeout_prob: float = 0.5,
    c_low_conf_prob: float = 0.5,
) -> SpecialistRunner:
    return SpecialistRunner(
        rng=rng,
        b_timeout_prob=b_timeout_prob,
        c_low_conf_prob=c_low_conf_prob,
        use_real=False,
    )


def make_real_runner(
    rng: random.Random | None = None,
    b_timeout_prob: float = 0.5,
    c_low_conf_prob: float = 0.5,
) -> SpecialistRunner:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required for --real mode.")
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq

    client = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
    return SpecialistRunner(
        rng=rng,
        b_timeout_prob=b_timeout_prob,
        c_low_conf_prob=c_low_conf_prob,
        use_real=True,
        groq_client=client,
    )


# ---------------------------------------------------------------------------
# LangGraph State & Nodes
# ---------------------------------------------------------------------------
class SupervisorState(TypedDict, total=False):
    original_task: str
    subtasks: list[dict[str, str]]
    current_subtask_index: int
    current_tried_specialists: list[str]
    last_outcome: str
    subtask_results: Annotated[list[SpecialistResult], operator.add]
    degraded_subtasks: Annotated[list[str], operator.add]
    delegation_log: Annotated[list[DelegationDecision], operator.add]
    log: Annotated[list[str], operator.add]
    report: SupervisorReport


def decompose_node(state: SupervisorState) -> dict[str, Any]:
    task = state["original_task"]
    subtasks = [
        {
            "subtask": f"Phase 1 - System Architecture & Interface Specs: {task}",
            "primary_specialist": "Specialist A",
        },
        {
            "subtask": f"Phase 2 - Core Engine & Async Pipeline Implementation: {task}",
            "primary_specialist": "Specialist B",
        },
        {
            "subtask": f"Phase 3 - Security Audit, QA & Reliability Testing: {task}",
            "primary_specialist": "Specialist C",
        },
    ]
    return {
        "subtasks": subtasks,
        "current_subtask_index": 0,
        "current_tried_specialists": [],
        "last_outcome": "",
        "log": ["log: supervisor: decomposed task into 3 subtasks."],
    }


def make_attempt_node(runner: SpecialistRunner) -> Callable[[SupervisorState], dict[str, Any]]:
    def attempt_node(state: SupervisorState) -> dict[str, Any]:
        idx = state["current_subtask_index"]
        subtask_info = state["subtasks"][idx]
        subtask_text = subtask_info["subtask"]
        primary = subtask_info["primary_specialist"]
        tried = state.get("current_tried_specialists", [])

        if primary not in tried:
            candidate = primary
        else:
            untried = [s for s in ALL_SPECIALISTS if s not in tried]
            candidate = untried[0]

        attempt_num = len(tried) + 1

        try:
            res = runner.run_specialist(candidate, subtask_text)
            if not res.succeeded or res.confidence < CONFIDENCE_THRESHOLD:
                outcome = "low_confidence"
                reasoning = (
                    f"Specialist '{candidate}' attempt #{attempt_num} returned low confidence "
                    f"({res.confidence:.2f} < {CONFIDENCE_THRESHOLD}). Rerouting required."
                )
                decision = DelegationDecision(
                    subtask=subtask_text,
                    specialist_name=candidate,
                    attempt_number=attempt_num,
                    outcome=outcome,
                    reasoning=reasoning,
                )
                return {
                    "current_tried_specialists": list(tried) + [candidate],
                    "last_outcome": outcome,
                    "delegation_log": [decision],
                    "log": [f"log: supervisor: {candidate} on subtask #{idx+1} returned low_confidence ({res.confidence:.2f})."],
                }
            else:
                outcome = "success"
                reasoning = (
                    f"Specialist '{candidate}' attempt #{attempt_num} succeeded with confidence {res.confidence:.2f}."
                )
                decision = DelegationDecision(
                    subtask=subtask_text,
                    specialist_name=candidate,
                    attempt_number=attempt_num,
                    outcome=outcome,
                    reasoning=reasoning,
                )
                return {
                    "subtask_results": [res],
                    "delegation_log": [decision],
                    "current_subtask_index": idx + 1,
                    "current_tried_specialists": [],
                    "last_outcome": outcome,
                    "log": [f"log: supervisor: {candidate} on subtask #{idx+1} succeeded."],
                }
        except Exception as err:
            outcome = "timeout"
            reasoning = (
                f"Specialist '{candidate}' attempt #{attempt_num} failed due to simulated timeout/error: {err}. "
                f"Rerouting required."
            )
            decision = DelegationDecision(
                subtask=subtask_text,
                specialist_name=candidate,
                attempt_number=attempt_num,
                outcome=outcome,
                reasoning=reasoning,
            )
            return {
                "current_tried_specialists": list(tried) + [candidate],
                "last_outcome": outcome,
                "delegation_log": [decision],
                "log": [f"log: supervisor: {candidate} on subtask #{idx+1} timed out."],
            }

    return attempt_node


def route_after_attempt(state: SupervisorState) -> Literal["attempt", "degrade", "aggregate"]:
    idx = state["current_subtask_index"]
    if idx >= len(state["subtasks"]):
        return "aggregate"

    last_outcome = state.get("last_outcome", "")
    if last_outcome == "success":
        return "attempt"

    tried = state.get("current_tried_specialists", [])
    if len(tried) >= len(ALL_SPECIALISTS):
        return "degrade"
    return "attempt"


def degrade_node(state: SupervisorState) -> dict[str, Any]:
    idx = state["current_subtask_index"]
    subtask_info = state["subtasks"][idx]
    subtask_text = subtask_info["subtask"]
    tried = state.get("current_tried_specialists", [])
    attempt_num = len(tried)

    placeholder = SpecialistResult(
        subtask=subtask_text,
        specialist_name="None (Exhausted)",
        content=f"[DEGRADED] Placeholder result: all {attempt_num} specialist attempts failed for this subtask.",
        confidence=0.0,
        succeeded=False,
    )
    decision = DelegationDecision(
        subtask=subtask_text,
        specialist_name="All Specialists",
        attempt_number=attempt_num,
        outcome="exhausted",
        reasoning=(
            f"Exhausted all {attempt_num} specialists ({', '.join(tried)}) for subtask: '{subtask_text}'. "
            f"Applying graceful degradation placeholder."
        ),
    )
    return {
        "subtask_results": [placeholder],
        "degraded_subtasks": [subtask_text],
        "delegation_log": [decision],
        "current_subtask_index": idx + 1,
        "current_tried_specialists": [],
        "last_outcome": "exhausted",
        "log": [f"log: supervisor: degraded subtask #{idx+1} after all {attempt_num} attempts failed."],
    }


def route_after_degrade(state: SupervisorState) -> Literal["attempt", "aggregate"]:
    idx = state["current_subtask_index"]
    if idx >= len(state["subtasks"]):
        return "aggregate"
    return "attempt"


def aggregate_node(state: SupervisorState) -> dict[str, Any]:
    original_task = state["original_task"]
    results = state.get("subtask_results", [])
    degraded = state.get("degraded_subtasks", [])
    log_decisions = state.get("delegation_log", [])

    status: Literal["complete", "degraded"] = "degraded" if len(degraded) > 0 else "complete"

    report = SupervisorReport(
        original_task=original_task,
        subtask_results=results,
        degraded_subtasks=degraded,
        delegation_log=log_decisions,
        overall_status=status,
    )
    return {
        "report": report,
        "log": [f"log: supervisor: aggregated report with status '{status}'."],
    }


def build_graph(runner: SpecialistRunner | None = None) -> StateGraph:
    if runner is None:
        runner = make_stub_runner()

    workflow = StateGraph(SupervisorState)

    workflow.add_node("decompose", decompose_node)
    workflow.add_node("attempt", make_attempt_node(runner))
    workflow.add_node("degrade", degrade_node)
    workflow.add_node("aggregate", aggregate_node)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "attempt")

    workflow.add_conditional_edges(
        "attempt",
        route_after_attempt,
        {
            "attempt": "attempt",
            "degrade": "degrade",
            "aggregate": "aggregate",
        },
    )

    workflow.add_conditional_edges(
        "degrade",
        route_after_degrade,
        {
            "attempt": "attempt",
            "aggregate": "aggregate",
        },
    )

    workflow.add_edge("aggregate", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
@app.command()
def main(
    task: str = typer.Argument(..., help="Complex task description to decompose and delegate."),
    real: bool = typer.Option(False, "--real", help="Use real Groq LLM for content synthesis."),
    seed: int | None = typer.Option(None, "--seed", help="Optional random seed for reproducible failures."),
) -> None:
    rng = random.Random(seed) if seed is not None else random.Random()

    if real:
        try:
            runner = make_real_runner(rng=rng)
        except RuntimeError as err:
            console.print(f"[bold red]Error:[/bold red] {err}")
            raise typer.Exit(code=1)
    else:
        runner = make_stub_runner(rng=rng)

    graph = build_graph(runner=runner)
    result = graph.invoke({
        "original_task": task,
        "log": [],
        "subtask_results": [],
        "degraded_subtasks": [],
        "delegation_log": [],
    })

    report: SupervisorReport = result["report"]

    # Render Delegation Log Table
    table = Table(title=f"Supervisor Delegation Log  ·  Task: '{task}'")
    table.add_column("Subtask", overflow="fold")
    table.add_column("Specialist")
    table.add_column("Attempt #", justify="center")
    table.add_column("Outcome")
    table.add_column("Reasoning", overflow="fold")

    outcome_styles = {
        "success": "green",
        "timeout": "bold red",
        "low_confidence": "yellow",
        "exhausted": "bold red reverse",
    }

    for decision in report.delegation_log:
        style = outcome_styles.get(decision.outcome, "white")
        table.add_row(
            decision.subtask,
            decision.specialist_name,
            str(decision.attempt_number),
            f"[{style}]{decision.outcome}[/{style}]",
            decision.reasoning,
        )

    console.print(table)
    console.print()

    # Render Subtask Panels
    for res in report.subtask_results:
        if res.succeeded:
            panel_style = "green"
            status_text = "[bold green]SUCCESS[/bold green]"
        else:
            panel_style = "yellow"
            status_text = "[bold red]DEGRADED[/bold red]"

        body = (
            f"{res.content}\n\n"
            f"Confidence: [bold]{res.confidence:.2f}[/bold] | Status: {status_text}"
        )
        console.print(
            Panel(
                body,
                title=f"Subtask: {res.subtask}",
                subtitle=f"Specialist: {res.specialist_name}",
                style=panel_style,
            )
        )

    # Render Summary Panel
    summary_style = "green" if report.overall_status == "complete" else "yellow"
    summary_text = (
        f"Overall Status: [bold]{report.overall_status.upper()}[/bold]\n"
        f"Total Subtasks: {len(report.subtask_results)}  ·  Degraded Subtasks: {len(report.degraded_subtasks)}"
    )
    console.print(Panel(summary_text, title="Supervisor Executive Summary", style=summary_style))


if __name__ == "__main__":
    app()
