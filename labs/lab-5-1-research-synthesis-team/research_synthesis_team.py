"""
Week 5, Day 1 (Monday) -- First Multi-Agent Team: Research + Synthesis
CalderR Agentic AI Engineering Internship, Week 5

    research_agent -> synthesis_agent

This is deliberately the simplest possible multi-agent shape: a fixed
two-step pipeline where the edge never branches and no node decides
*at runtime* who acts next. That's the point of building it first --
CAMEL's two-agent role-play, AutoGen's two-agent assistant/user-proxy
chat, and AgentVerse's vertical structure (solver + reviewers) are all
more elaborate versions of the same underlying idea: one agent produces
something, another agent consumes it. What actually separates a
"pipeline" from an "orchestrator" or "supervisor" isn't the agent
count, it's whether a node inspects the state and *chooses* the next
agent (a conditional edge, like `route_after_validate` in lab-4-1, or a
supervisor picking a specialist based on the task) versus a fixed edge
that always goes the same place. Tuesday's Supervisor lab adds exactly
that decision; this lab intentionally doesn't have one yet.

The handoff between the two agents is the actual assignment: everything
the Research Agent finds is packaged into one typed Pydantic object
(ResearchHandoff) before the Synthesis Agent ever sees it, instead of
passing a raw string or an untyped dict across the boundary. Consider
this a one-message preview of what Lab 5.1 (typed message bus) turns
into a reusable pattern later this week.

Usage:
    python research_synthesis_team.py "langgraph"
    python research_synthesis_team.py "multi-agent systems"
    python research_synthesis_team.py "quantum computing"   # deliberate miss --
        not indexed anywhere, shows a real coverage gap surfacing honestly
        instead of the report papering over it
    python research_synthesis_team.py "groq" --real

(Typer collapses to a single positional argument -- TOPIC -- when only one
command is registered, so there's no subcommand word to type here; only
--real is a flag. Run with --help to see this reflected in the usage line.)

--real switches the Synthesis Agent from the offline stub (rule-based,
zero network calls) to an actual Groq call (llama-3.3-70b-versatile via
langchain-groq), matching the rest of the repo's stack. Requires
GROQ_API_KEY in the environment. Default is the stub, so this file runs
end-to-end with zero network access and zero credentials required --
see smoke_test.py.
"""

from __future__ import annotations

import operator
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Callable, Literal, TypedDict

import typer
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

app = typer.Typer()
console = Console()


# ---------------------------------------------------------------------------
# Typed inter-agent messages
# ---------------------------------------------------------------------------
# These three models are what the brief means by "type all inter-agent
# messages with Pydantic." ResearchHandoff in particular is the message
# that actually crosses the agent boundary -- the Synthesis Agent never
# receives a raw string, only a validated object with a known shape.
class SourceFinding(BaseModel):
    """One finding pulled from a single mock source for a single query."""

    source_type: Literal["web", "academic", "internal_docs"]
    source_name: str
    query_used: str
    content: str
    relevance: Literal["high", "medium", "low"] = "medium"


class ResearchHandoff(BaseModel):
    """The Research Agent's complete typed handoff to the Synthesis Agent --
    every source it queried, what it found (or didn't), and a short note
    on coverage gaps. The Synthesis Agent should need nothing beyond this
    object to do its job."""

    topic: str
    findings: list[SourceFinding]
    sources_queried: list[str]
    sources_with_no_results: list[str] = Field(default_factory=list)
    research_notes: str = ""


class SynthesisReport(BaseModel):
    """Final structured output from the Synthesis Agent."""

    topic: str
    title: str
    executive_summary: str = Field(
        description="2-3 sentences summarising what was found, grounded only in the handoff."
    )
    key_findings: list[str] = Field(
        description="3-6 standalone bullet points, each attributable to a source in the handoff."
    )
    sources_used: list[str]
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence given how many sources actually had coverage."
    )
    caveats: list[str] = Field(default_factory=list)
    report_id: str = ""
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Mock sources ("mock tools")
# ---------------------------------------------------------------------------
# Three independent mini knowledge-bases standing in for real retrieval
# (a web search API, an academic index, an internal wiki). Deliberately
# uneven coverage -- internal_docs only knows about two topics -- because
# a Researcher hitting a real gap in one source and still delivering a
# useful (lower-confidence) report is more realistic, and more useful to
# demo, than three sources that always agree.
WEB_SOURCES: dict[str, str] = {
    "multi-agent systems": (
        "Engineering blogs increasingly describe a shift from one monolithic "
        "agent toward small teams of specialized agents that hand off typed "
        "messages, echoing how software teams split work by role instead of "
        "having one generalist do everything."
    ),
    "langgraph": (
        "LangGraph's node-and-edge model lets a workflow's shape change based "
        "on runtime data, unlike a fixed linear chain. Conditional edges and "
        "checkpointer-backed persistence are the two features teams reach for "
        "first when moving agents into production."
    ),
    "groq": (
        "Groq's LPU (Language Processing Unit) hardware targets fast, "
        "low-latency token generation rather than training, which is why "
        "teams building responsive multi-turn agents often reach for it over "
        "GPU-based inference for interactive workloads."
    ),
    "pakistan ai policy": (
        "Pakistan's National Data Governance Policy 2026 sets rules for where "
        "public-sector data may be processed, pushing government-facing AI "
        "vendors toward locally-deployable systems rather than foreign-hosted "
        "APIs for sensitive workloads."
    ),
    "llm evaluation": (
        "Teams evaluating LLM agents increasingly track per-step metrics -- "
        "tool-call accuracy, handoff correctness, latency per node -- rather "
        "than only scoring the final answer, since a wrong intermediate step "
        "can still produce a plausible-looking final output."
    ),
}

ACADEMIC_SOURCES: dict[str, str] = {
    "multi-agent systems": (
        "Recent multi-agent LLM papers converge on three recurring design "
        "questions: how roles are assigned, how agents hand off state, and "
        "how disagreement gets resolved. Frameworks mainly differ in whether "
        "that composition is fixed at design time or adjusted dynamically "
        "at runtime based on feedback."
    ),
    "langgraph": (
        "Graph-based agent orchestration is typically framed as a state "
        "machine: nodes compute, edges route, and a conditional edge lets "
        "routing depend on data produced during execution rather than being "
        "fixed in advance."
    ),
    "groq": (
        "Hardware specialised for inference rather than general matrix "
        "multiplication trades some flexibility for markedly lower "
        "per-token latency, which matters disproportionately for agent "
        "architectures where one user-facing task can trigger many "
        "sequential LLM calls."
    ),
    "consensus mechanisms": (
        "Confidence-weighted voting among agents tends to outperform simple "
        "majority vote when individual agents can report calibrated "
        "confidence, because a low-confidence wrong answer gets "
        "down-weighted instead of counting equally with a high-confidence "
        "correct one."
    ),
}

INTERNAL_DOCS_SOURCES: dict[str, str] = {
    # Sparse on purpose -- an internal knowledge base realistically won't
    # cover everything a Researcher goes looking for. That gap is exactly
    # what ResearchHandoff.sources_with_no_results exists to surface.
    "langgraph": (
        "Internal eng wiki: the repo standard for new agent graphs is "
        "TypedDict state with an append-only 'log' field "
        "(Annotated[list[str], operator.add]) for an audit trail, plus a "
        "stub/real toggle on every node that calls an external API."
    ),
    "groq": (
        "Internal platform note: GROQ_API_KEY is provisioned via the shared "
        "secrets vault, never committed to a repo -- see the onboarding doc "
        "for the request process."
    ),
}


def _keyword_lookup(db: dict[str, str], query: str) -> tuple[str, bool]:
    """Shared lookup behind all three mock sources: exact key/word overlap
    first, then a looser partial match, then an honest miss. Returns
    (content, found) so callers get a real boolean instead of having to
    string-sniff a miss message."""
    q = query.lower()
    for key, value in db.items():
        if key in q or any(word in q for word in key.split()):
            return value, True
    for key, value in db.items():
        if any(word in key for word in q.split() if len(word) > 3):
            return value, True
    available = ", ".join(sorted(db.keys()))
    return f"No matching entries for '{query}'. Indexed topics: {available}.", False


@tool
def mock_web_search(query: str) -> str:
    """Search general web results for a topic. Input: a short query."""
    content, _ = _keyword_lookup(WEB_SOURCES, query)
    return content


@tool
def mock_academic_search(query: str) -> str:
    """Search an academic/paper index for a topic. Input: a short query."""
    content, _ = _keyword_lookup(ACADEMIC_SOURCES, query)
    return content


@tool
def mock_internal_docs_search(query: str) -> str:
    """Search internal company documentation for a topic. Input: a short query."""
    content, _ = _keyword_lookup(INTERNAL_DOCS_SOURCES, query)
    return content


# (source_type, display name, backing db) -- the Research Agent below
# queries all three deterministically. Nothing here decides *which*
# source to call via an LLM; ReAct-style tool selection was Week 2's
# lesson, this lab's lesson is the typed handoff that comes after.
_SOURCES: list[tuple[str, str, dict[str, str]]] = [
    ("web", "Web Index", WEB_SOURCES),
    ("academic", "Academic Digest", ACADEMIC_SOURCES),
    ("internal_docs", "Internal Knowledge Base", INTERNAL_DOCS_SOURCES),
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class TeamState(TypedDict, total=False):
    topic: str
    research_handoff: ResearchHandoff
    report: SynthesisReport
    # Annotated + operator.add makes this an append-only reducer -- both
    # agents contribute to one growing audit trail instead of each node's
    # return value overwriting the last node's, same pattern as lab-4-3.
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Agent 1: Research Agent
# ---------------------------------------------------------------------------
def research_node(state: TeamState) -> dict:
    """Query all three mock sources for the topic and package the results
    into one typed ResearchHandoff. This node makes no LLM call at all --
    it doesn't need one to demonstrate the handoff, and keeping it plain
    Python means the Researcher's half of this lab runs identically
    whether or not GROQ_API_KEY is set."""
    topic = state["topic"]
    findings: list[SourceFinding] = []
    sources_queried: list[str] = []
    sources_with_no_results: list[str] = []

    for source_type, source_name, db in _SOURCES:
        content, found = _keyword_lookup(db, topic)
        sources_queried.append(source_name)
        if not found:
            sources_with_no_results.append(source_name)
        findings.append(
            SourceFinding(
                source_type=source_type,
                source_name=source_name,
                query_used=topic,
                content=content,
                relevance="high" if found else "low",
            )
        )

    notes = (
        "All sources returned an indexed match."
        if not sources_with_no_results
        else f"{len(sources_with_no_results)} of {len(sources_queried)} sources had no indexed match."
    )

    handoff = ResearchHandoff(
        topic=topic,
        findings=findings,
        sources_queried=sources_queried,
        sources_with_no_results=sources_with_no_results,
        research_notes=notes,
    )
    return {
        "research_handoff": handoff,
        "log": [f"research_agent: queried {len(sources_queried)} sources for '{topic}' -- {notes}"],
    }


# ---------------------------------------------------------------------------
# Agent 2: Synthesis Agent -- stub vs real, same pattern as lab-4-3
# ---------------------------------------------------------------------------
def make_stub_synthesizer() -> Callable[[ResearchHandoff], SynthesisReport]:
    """Deterministic, zero-network synthesis -- exercises the full
    handoff -> report shape without an LLM or API key. Swap in
    make_real_synthesizer() for actual language generation."""

    def synthesize(handoff: ResearchHandoff) -> SynthesisReport:
        hits = [f for f in handoff.findings if f.relevance != "low"]
        key_findings = [f"[{f.source_name}] {f.content}" for f in hits] or [
            "No indexed source had a match for this topic."
        ]

        queried, missed = len(handoff.sources_queried), len(handoff.sources_with_no_results)
        confidence: Literal["high", "medium", "low"]
        if missed == 0:
            confidence = "high"
        elif missed < queried / 2:
            confidence = "medium"
        else:
            confidence = "low"

        caveats = []
        if handoff.sources_with_no_results:
            caveats.append(f"No coverage from: {', '.join(handoff.sources_with_no_results)}.")

        return SynthesisReport(
            topic=handoff.topic,
            title=f"Research Briefing: {handoff.topic.title()}",
            executive_summary=(
                f"Synthesised from {len(hits)} of {len(handoff.findings)} queried sources. "
                f"{handoff.research_notes}"
            ),
            key_findings=key_findings,
            sources_used=[f.source_name for f in hits],
            confidence=confidence,
            caveats=caveats,
            report_id=str(uuid.uuid4())[:8],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    return synthesize


def make_real_synthesizer() -> Callable[[ResearchHandoff], SynthesisReport]:
    """Groq-backed synthesis -- matches the rest of the repo's stack
    (llama-3.3-70b-versatile via ChatGroq). Requires GROQ_API_KEY."""
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it, or drop --real to use "
            "the offline stub synthesizer instead."
        )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(
        SynthesisReport
    )

    def synthesize(handoff: ResearchHandoff) -> SynthesisReport:
        prompt = (
            "You are a Synthesis Agent. A Research Agent already queried "
            "multiple sources and is handing you its typed findings below -- "
            "do not invent facts beyond what's in this handoff. Write a "
            "structured research briefing: a short title, a 2-3 sentence "
            "executive summary, 3-6 key findings as separate strings, which "
            "sources you actually drew on, an honest confidence level given "
            "any source gaps, and any caveats worth flagging.\n\n"
            f"Research handoff (JSON):\n{handoff.model_dump_json(indent=2)}"
        )
        result = llm.invoke(prompt)
        # topic/report_id/generated_at shouldn't come from the model's
        # imagination -- topic must match the handoff exactly, and an id
        # or timestamp the LLM invents isn't a real id or timestamp.
        return result.model_copy(
            update={
                "topic": handoff.topic,
                "report_id": str(uuid.uuid4())[:8],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return synthesize


def make_synthesis_node(synthesizer: Callable[[ResearchHandoff], SynthesisReport]):
    def synthesis_node(state: TeamState) -> dict:
        report = synthesizer(state["research_handoff"])
        return {
            "report": report,
            "log": [f"synthesis_agent: produced '{report.title}' (confidence={report.confidence})"],
        }

    return synthesis_node


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(synthesizer: Callable[[ResearchHandoff], SynthesisReport]):
    builder = StateGraph(TeamState)
    builder.add_node("research_agent", research_node)
    builder.add_node("synthesis_agent", make_synthesis_node(synthesizer))

    # Fixed edges only -- no conditional routing. See the module docstring:
    # this is what makes it a pipeline rather than an orchestrator.
    builder.add_edge(START, "research_agent")
    builder.add_edge("research_agent", "synthesis_agent")
    builder.add_edge("synthesis_agent", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def research(
    topic: str,
    real: bool = typer.Option(
        False, "--real", help="Use the Groq-backed Synthesis Agent instead of the offline stub."
    ),
):
    """Run TOPIC through Research Agent -> Synthesis Agent and print both
    the typed handoff and the final structured report."""
    try:
        synthesizer = make_real_synthesizer() if real else make_stub_synthesizer()
    except RuntimeError as exc:
        # Caught here, inside the command callback, so typer.Exit below is
        # still within Click's managed dispatch and exits cleanly -- no
        # traceback. Catching this same error one level up around app()
        # instead would be too late: by the time it propagated there,
        # we'd already be past Click's own exception handling, and a bare
        # `raise typer.Exit(...)` at that point just becomes an ordinary
        # unhandled exception with a full traceback dumped to stderr.
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    graph = build_graph(synthesizer)
    result = graph.invoke({"topic": topic, "log": []})

    handoff: ResearchHandoff = result["research_handoff"]
    report: SynthesisReport = result["report"]

    table = Table(title=f"ResearchHandoff  ·  topic='{topic}'")
    table.add_column("Source")
    table.add_column("Relevance")
    table.add_column("Content", overflow="fold")
    for f in handoff.findings:
        style = "dim" if f.relevance == "low" else None
        table.add_row(f.source_name, f.relevance, f.content, style=style)
    console.print(table)
    console.print(f"[dim]{handoff.research_notes}[/dim]\n")

    confidence_style = {"high": "green", "medium": "yellow", "low": "red"}[report.confidence]
    body = report.executive_summary + "\n\n" + "\n".join(f"- {kf}" for kf in report.key_findings)
    if report.caveats:
        body += "\n\n[yellow]Caveats:[/yellow]\n" + "\n".join(f"- {c}" for c in report.caveats)

    console.print(
        Panel(
            body,
            title=f"{report.title}  ·  confidence: {report.confidence}",
            subtitle=f"sources: {', '.join(report.sources_used) or 'none'}  ·  id={report.report_id}",
            style=confidence_style,
        )
    )

    for line in result.get("log", []):
        console.print(f"[dim]log: {line}[/dim]")


if __name__ == "__main__":
    app()
