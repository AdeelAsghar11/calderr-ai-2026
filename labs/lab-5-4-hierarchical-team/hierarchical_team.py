"""
Week 5, Day 4 (Applied Practice) -- Hierarchical Multi-Agent Software Engineering Team
CalderR Agentic AI Engineering Internship, Week 5

Tier 1 (Executive)     PM agent
                            |
                +-----------+-----------+
                |                       |
Tier 2 (Leads)  Engineering lead        QA lead
                |          |            |            |
Tier 3 (Workers) Backend  Frontend   Test writer  Test executor

In this lab, we build a 3-tier hierarchical multi-agent system where state flows strictly
across typed boundaries without leaking internal agent context.

Why "hierarchical" is tool-wrapping:
A "hierarchical" team in LangGraph / LangChain is simply the exact same tool-wrapping mechanism
applied recursively. A Lead is not a fundamentally different graph node type; it is an agent
whose tools happen to invoke other sub-agents. By having each sub-agent invoked inside a tool
wrapper that returns only its final text or typed result (and not the full parent message history
or sub-agent internal reasoning/tool calls), we guarantee that context remains strictly scoped to
its respective tier.

Usage:
    uv run python labs/lab-5-4-hierarchical-team/hierarchical_team.py "Build user authentication service"
    uv run python labs/lab-5-4-hierarchical-team/hierarchical_team.py "Build payments gateway" --real
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool
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


# ---------------------------------------------------------------------------
# Typed inter-agent & report models
# ---------------------------------------------------------------------------
class BuildSummary(BaseModel):
    """Aggregated result returned by the Engineering Lead to the PM."""

    components_built: list[str] = Field(
        description="List of system components constructed."
    )
    backend_summary: str = Field(description="Summary of backend work completed.")
    frontend_summary: str = Field(description="Summary of frontend work completed.")
    issues: list[str] = Field(
        default_factory=list, description="Technical debt or issues surfaced."
    )


class QASummary(BaseModel):
    """Aggregated result returned by the QA Lead to the PM."""

    test_cases_written: int = Field(description="Number of test cases written.")
    tests_passed: int = Field(description="Number of tests that passed.")
    tests_failed: int = Field(description="Number of tests that failed.")
    failures: list[str] = Field(
        default_factory=list, description="Descriptions of any test failures."
    )


class ReleaseReport(BaseModel):
    """Final release decision and report synthesized by the PM agent."""

    feature: str = Field(description="The feature or requirement brief.")
    build_summary: BuildSummary
    qa_summary: QASummary
    overall_status: Literal["ready", "blocked"] = Field(
        description="'blocked' if tests_failed > 0, otherwise 'ready'."
    )
    release_notes: str = Field(
        description="Synthesized release notes for stakeholders."
    )


# ---------------------------------------------------------------------------
# Helper JSON extraction
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict[str, Any]:
    """Helper to extract JSON dictionary from LLM string output."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            text = text[first_brace : last_brace + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Offline Stub Implementation (Default)
# ---------------------------------------------------------------------------
# The stub implementation matches the exact 3-tier boundary guarantees of the
# real agent graph. Each agent is a deterministic function that maintains its own
# private internal scratchpad/state, passing only typed summaries across boundaries.


def stub_backend_agent(requirements: str, internal_marker: str = "") -> dict[str, Any]:
    """Tier 3 Worker: Backend Agent (Stub)."""
    scratchpad = (
        f"[Backend Internal Reasoning] Parsed requirements '{requirements}'. "
        f"Designing DB schema and API routes. {internal_marker}".strip()
    )
    result_summary = f"Built REST API endpoints and data models for '{requirements}'."
    return {
        "summary": result_summary,
        "component": "backend_api",
        "internal_scratchpad": scratchpad,
    }


def stub_frontend_agent(requirements: str) -> dict[str, Any]:
    """Tier 3 Worker: Frontend Agent (Stub)."""
    scratchpad = f"[Frontend Internal Reasoning] Designing UI layout for '{requirements}'."
    result_summary = f"Built responsive UI component views for '{requirements}'."
    return {
        "summary": result_summary,
        "component": "frontend_ui",
        "internal_scratchpad": scratchpad,
    }


def stub_engineering_lead(
    requirements: str, backend_marker: str = ""
) -> tuple[BuildSummary, dict[str, Any]]:
    """Tier 2 Lead: Engineering Lead (Stub).

    Calls Backend and Frontend workers, receives their outputs, and aggregates
    ONLY public summaries into BuildSummary. Internal worker scratchpads are omitted.
    """
    backend_res = stub_backend_agent(requirements, internal_marker=backend_marker)
    frontend_res = stub_frontend_agent(requirements)

    build_summary = BuildSummary(
        components_built=[backend_res["component"], frontend_res["component"]],
        backend_summary=backend_res["summary"],
        frontend_summary=frontend_res["summary"],
        issues=[],
    )

    lead_internal_state = {
        "subtask_outputs": [backend_res["summary"], frontend_res["summary"]],
        "aggregated_build": build_summary.model_dump(),
    }
    return build_summary, lead_internal_state


def stub_test_writer_agent(requirements: str) -> dict[str, Any]:
    """Tier 3 Worker: Test Writer Agent (Stub)."""
    scratchpad = f"[Test Writer Internal Reasoning] Generating test cases for '{requirements}'."
    test_cases = [
        f"Test happy path for {requirements}",
        f"Test edge cases & error handling for {requirements}",
    ]
    return {
        "test_cases": test_cases,
        "count": len(test_cases),
        "internal_scratchpad": scratchpad,
    }


def stub_test_executor_agent(
    build_summary: BuildSummary, test_cases: list[str]
) -> dict[str, Any]:
    """Tier 3 Worker: Test Executor Agent (Stub)."""
    scratchpad = f"[Test Executor Internal Reasoning] Executing {len(test_cases)} tests against components: {build_summary.components_built}."
    # Determine pass/fail based on presence of explicit failure keywords or issues
    has_issues = len(build_summary.issues) > 0
    passed = len(test_cases) if not has_issues else max(0, len(test_cases) - 1)
    failed = 0 if not has_issues else 1
    failures = [] if not has_issues else ["Encountered technical debt issue in build"]

    return {
        "passed": passed,
        "failed": failed,
        "failures": failures,
        "internal_scratchpad": scratchpad,
    }


def stub_qa_lead(
    build_summary: BuildSummary, requirements: str
) -> tuple[QASummary, dict[str, Any]]:
    """Tier 2 Lead: QA Lead (Stub).

    Calls Test Writer and Test Executor workers. Aggregates into QASummary.
    Never inspects Engineering's private state.
    """
    tw_res = stub_test_writer_agent(requirements)
    te_res = stub_test_executor_agent(build_summary, tw_res["test_cases"])

    qa_summary = QASummary(
        test_cases_written=tw_res["count"],
        tests_passed=te_res["passed"],
        tests_failed=te_res["failed"],
        failures=te_res["failures"],
    )

    lead_internal_state = {
        "test_cases_count": tw_res["count"],
        "qa_summary": qa_summary.model_dump(),
    }
    return qa_summary, lead_internal_state


def stub_pm_agent(
    requirements: str, backend_marker: str = ""
) -> tuple[ReleaseReport, dict[str, Any]]:
    """Tier 1 Executive: PM Agent (Stub).

    Delegates to Engineering Lead and QA Lead sequentially. PM never sees worker
    scratchpads or Lead intermediate states—only typed summaries.
    """
    build_summary, eng_state = stub_engineering_lead(
        requirements, backend_marker=backend_marker
    )
    qa_summary, qa_state = stub_qa_lead(build_summary, requirements)

    # Compute overall status strictly in code based on tests_failed rule
    overall_status: Literal["ready", "blocked"] = (
        "blocked" if qa_summary.tests_failed > 0 else "ready"
    )

    release_notes = (
        f"Feature '{requirements}' successfully built and verified. "
        f"Components: {', '.join(build_summary.components_built)}. "
        f"QA Results: {qa_summary.tests_passed}/{qa_summary.test_cases_written} tests passed."
    )

    report = ReleaseReport(
        feature=requirements,
        build_summary=build_summary,
        qa_summary=qa_summary,
        overall_status=overall_status,
        release_notes=release_notes,
    )

    pm_internal_state = {
        "assigned_engineering": True,
        "assigned_qa": True,
        "release_report": report.model_dump(),
    }

    full_trace = {
        "backend_worker_scratchpad": stub_backend_agent(
            requirements, internal_marker=backend_marker
        )["internal_scratchpad"],
        "engineering_lead_state": eng_state,
        "qa_lead_state": qa_state,
        "pm_state": pm_internal_state,
    }

    return report, full_trace


# ---------------------------------------------------------------------------
# Real Groq LLM Agent Implementation (--real)
# ---------------------------------------------------------------------------
def run_real_hierarchical_team(requirements: str) -> ReleaseReport:
    """Runs the 3-tier hierarchical team using real Groq LLM agents and tool wrapping."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is missing. Set it or run without --real for stub mode."
        )

    # pyrefly: ignore [missing-import]
    from langchain.agents import create_agent
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # --- Tier 3 Workers ---
    backend_agent = create_agent(
        model=llm,
        system_prompt="You are a Backend Engineer. Return a concise summary of backend work done for the given feature brief.",
    )
    frontend_agent = create_agent(
        model=llm,
        system_prompt="You are a Frontend Engineer. Return a concise summary of frontend work done for the given feature brief.",
    )
    test_writer_agent = create_agent(
        model=llm,
        system_prompt="You are a QA Test Writer. Output test cases as a JSON list of strings.",
    )
    test_executor_agent = create_agent(
        model=llm,
        system_prompt="You are a QA Test Executor. Execute tests and output a JSON with 'passed' (int), 'failed' (int), and 'failures' (list of strings).",
    )

    # --- Tool Wrappers for Tier 3 Workers (Used by Tier 2 Leads) ---
    @tool
    def assign_backend_worker(brief: str) -> str:
        """Assign backend task to Backend Worker."""
        res = backend_agent.invoke({"messages": [("user", f"Build backend for: {brief}")]})
        return str(res["messages"][-1].content)

    @tool
    def assign_frontend_worker(brief: str) -> str:
        """Assign frontend task to Frontend Worker."""
        res = frontend_agent.invoke({"messages": [("user", f"Build frontend for: {brief}")]})
        return str(res["messages"][-1].content)

    @tool
    def write_test_cases(brief: str) -> str:
        """Generate test cases for the feature brief."""
        res = test_writer_agent.invoke({"messages": [("user", f"Write tests for: {brief}")]})
        return str(res["messages"][-1].content)

    @tool
    def execute_test_cases(build_summary_str: str, test_cases_str: str) -> str:
        """Execute test cases against the build summary."""
        res = test_executor_agent.invoke(
            {"messages": [("user", f"Build: {build_summary_str}\nTests: {test_cases_str}")]}
        )
        return str(res["messages"][-1].content)

    # --- Tier 2 Leads ---
    eng_lead_agent = create_agent(
        model=llm,
        tools=[assign_backend_worker, assign_frontend_worker],
        system_prompt=(
            "You are Engineering Lead. Call assign_backend_worker and assign_frontend_worker. "
            "Return ONLY a raw JSON object matching BuildSummary schema: "
            '{"components_built": [...], "backend_summary": "...", "frontend_summary": "...", "issues": []}'
        ),
    )

    qa_lead_agent = create_agent(
        model=llm,
        tools=[write_test_cases, execute_test_cases],
        system_prompt=(
            "You are QA Lead. Call write_test_cases and execute_test_cases. "
            "Return ONLY a raw JSON object matching QASummary schema: "
            '{"test_cases_written": int, "tests_passed": int, "tests_failed": int, "failures": []}'
        ),
    )

    # --- Tool Wrappers for Tier 2 Leads (Used by Tier 1 Executive) ---
    @tool
    def assign_engineering(brief: str) -> str:
        """Assign software construction to Engineering Lead. Returns BuildSummary JSON string."""
        res = eng_lead_agent.invoke({"messages": [("user", f"Construct feature: {brief}")]})
        return str(res["messages"][-1].content)

    @tool
    def assign_qa(build_summary_json: str, brief: str) -> str:
        """Assign QA testing to QA Lead. Returns QASummary JSON string."""
        res = qa_lead_agent.invoke(
            {"messages": [("user", f"Validate build: {build_summary_json} for feature: {brief}")]}
        )
        return str(res["messages"][-1].content)

    # --- Tier 1 Executive ---
    pm_agent = create_agent(
        model=llm,
        tools=[assign_engineering, assign_qa],
        system_prompt=(
            "You are PM Agent. First call assign_engineering. Then call assign_qa with the returned BuildSummary. "
            "Synthesize into a final ReleaseReport JSON matching schema: "
            '{"feature": "...", "build_summary": {...}, "qa_summary": {...}, "overall_status": "ready", "release_notes": "..."}'
        ),
    )

    res = pm_agent.invoke({"messages": [("user", f"Coordinate release for feature: {requirements}")]})
    final_content = str(res["messages"][-1].content)
    raw_json = _extract_json(final_content)

    # Hydrate models & enforce status override rule in code
    build_sum = BuildSummary(**raw_json.get("build_summary", {}))
    qa_sum = QASummary(**raw_json.get("qa_summary", {}))
    overall_status: Literal["ready", "blocked"] = (
        "blocked" if qa_sum.tests_failed > 0 else "ready"
    )

    return ReleaseReport(
        feature=requirements,
        build_summary=build_sum,
        qa_summary=qa_sum,
        overall_status=overall_status,
        release_notes=raw_json.get("release_notes", f"Release notes for {requirements}"),
    )


# ---------------------------------------------------------------------------
# CLI & Visual Proof
# ---------------------------------------------------------------------------
@app.command()
def main(
    feature: str = typer.Argument(..., help="Feature/requirements brief to build & verify."),
    real: bool = typer.Option(False, "--real", help="Use real Groq LLM agents instead of stub mode."),
) -> None:
    """Executes the 3-tier hierarchical multi-agent team and displays the release decision."""
    console.print(
        Panel.fit(
            f"[bold blue]Hierarchical Multi-Agent Team[/bold blue]\n[dim]Feature Brief: '{feature}' | Mode: {'Real (Groq)' if real else 'Offline Stub'}[/dim]",
            border_style="blue",
        )
    )

    if real:
        report = run_real_hierarchical_team(feature)
    else:
        report, _ = stub_pm_agent(feature)

    # 1. Tier & Handoff Breakdown Table
    table = Table(title="Hierarchical Agent Hierarchy & Communication Flow", border_style="dim")
    table.add_column("Tier", style="bold cyan")
    table.add_column("Agent Name", style="bold yellow")
    table.add_column("Subordinates / Tools", style="magenta")
    table.add_column("Returned Upward (Boundary)", style="green")

    table.add_row(
        "Tier 1 (Executive)",
        "PM Agent",
        "Engineering Lead, QA Lead",
        f"ReleaseReport (Status: {report.overall_status.upper()})",
    )
    table.add_row(
        "Tier 2 (Lead)",
        "Engineering Lead",
        "Backend Worker, Frontend Worker",
        f"BuildSummary ({len(report.build_summary.components_built)} components)",
    )
    table.add_row(
        "Tier 2 (Lead)",
        "QA Lead",
        "Test Writer, Test Executor",
        f"QASummary ({report.qa_summary.tests_passed}/{report.qa_summary.test_cases_written} passed)",
    )
    table.add_row(
        "Tier 3 (Worker)",
        "Backend Worker",
        "None (Leaf)",
        f"Summary ({len(report.build_summary.backend_summary)} chars)",
    )
    table.add_row(
        "Tier 3 (Worker)",
        "Frontend Worker",
        "None (Leaf)",
        f"Summary ({len(report.build_summary.frontend_summary)} chars)",
    )
    table.add_row(
        "Tier 3 (Worker)",
        "Test Writer Worker",
        "None (Leaf)",
        f"{report.qa_summary.test_cases_written} Test Cases",
    )
    table.add_row(
        "Tier 3 (Worker)",
        "Test Executor Worker",
        "None (Leaf)",
        f"Test Results ({report.qa_summary.tests_passed} pass, {report.qa_summary.tests_failed} fail)",
    )

    console.print(table)

    # 2. Release Report Panel
    color = "green" if report.overall_status == "ready" else "red"
    report_md = f"""### Release Status: [{color}]{report.overall_status.upper()}[/{color}]

**Feature Brief:** {report.feature}

**Build Summary:**
- Components Built: {', '.join(report.build_summary.components_built)}
- Backend: {report.build_summary.backend_summary}
- Frontend: {report.build_summary.frontend_summary}
- Issues: {report.build_summary.issues if report.build_summary.issues else 'None'}

**QA Summary:**
- Tests Written: {report.qa_summary.test_cases_written}
- Passed: {report.qa_summary.tests_passed}
- Failed: {report.qa_summary.tests_failed}
- Failures: {report.qa_summary.failures if report.qa_summary.failures else 'None'}

**PM Release Notes:**
{report.release_notes}
"""

    console.print(
        Panel(
            report_md,
            title=f"Final Release Report [{report.overall_status.upper()}]",
            border_style=color,
        )
    )


if __name__ == "__main__":
    app()
