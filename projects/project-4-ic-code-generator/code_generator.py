"""
Project 4-I-C -- Iterative Code Generator
CalderR Agentic AI Engineering Internship, Week 4

generate -> execute -> [conditional: success / debug / give_up]
                            success -> success -> END
                            debug   -> debug -> generate   (cycle)
                            give_up -> give_up -> END

Execution is sandboxed: each attempt runs as its own subprocess, in an
isolated temp directory, under a wall-clock timeout, an RLIMIT_AS memory
cap, an RLIMIT_CPU cap, and (when the host supports it) inside a fresh
network namespace via `unshare --net` so a generated solution can't make
outbound calls even by accident. This is proportionate to "buggy
LLM-generated code," not adversarial-attacker-grade isolation --
full containers would be the next step up if that threat model changes.

debug_node does no LLM reasoning of its own -- consistent with lab 4.2's
validate_node philosophy (deterministic checking, not another judge
call), it just turns execute_node's raw pass/fail into a clean log
entry. The actual "fixing" intelligence lives in generate_fn's prompt,
which receives the previous code and the failure feedback.

Usage:
    python code_generator.py solve is_palindrome
    python code_generator.py run-all
Requires GROQ_API_KEY in the environment (via .env, matching the rest of
this repo). See smoke_test.py for a network-free check of the graph and
sandbox mechanics using scripted fake generators.
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import subprocess
import sys
import tempfile
try:
    import resource
except ImportError:
    resource = None
from typing import Annotated, Callable, Optional, TypedDict

import typer
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# pyrefly: ignore [missing-import]
from problems import PROBLEMS, PROBLEMS_BY_ID

load_dotenv()

app = typer.Typer()
console = Console()

MAX_ITERATIONS = 5
EXEC_TIMEOUT_SECONDS = 6.0

# See lab 4.2's self_correcting_loop.py: logging.basicConfig() is a
# no-op once another handler exists on the root logger, so attach a
# FileHandler directly to a named logger, and set propagate=False so
# messages don't also hit a root handler and print twice.
logger = logging.getLogger("code_generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.FileHandler("codegen_log.txt", mode="a")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


class CodeGenState(TypedDict, total=False):
    problem: dict
    code: str
    iteration: int
    max_iterations: int
    passed: bool
    last_error: str
    solved: bool
    gave_up: bool
    iterations_used: int
    # Annotated + operator.add: append-only audit trail across
    # generate/debug/success/give_up, same pattern as Lab 4.3's log field.
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------


def _network_isolation_available() -> bool:
    """One-time capability check, cached at import. `unshare --net`
    needs either root or unprivileged user namespaces enabled -- both
    common on real dev machines and CI, but not universal, so this
    degrades gracefully instead of assuming it'll work everywhere."""
    try:
        r = subprocess.run(["unshare", "--net", "true"], capture_output=True, timeout=3)
        return r.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


_HAS_NETNS = _network_isolation_available()


def _limit_resources():
    """Runs in the child process via preexec_fn, before exec. This is
    the backstop underneath the wall-clock timeout: a process that's
    still making syscalls in a tight loop can be slower to kill via
    timeout alone than via a hard CPU-time rlimit, and a memory-hungry
    solution should die on its own rather than pressuring the host."""
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _build_harness(code: str, function_name: str, test_cases: list[dict]) -> str:
    """Appends a small self-contained test runner after the generated
    code, so the whole thing can run as a single subprocess and report
    structured results on stdout behind a marker line.

    NOTE: deliberately not built with textwrap.dedent() over an
    f-string containing {code}. dedent() strips the *common* leading
    whitespace across every line of the final string -- but `code` is
    itself a complete, independently-indented multi-line Python block
    (def at column 0, body at column 4), and mixing it into a template
    that's indented for readability in *this* source file confuses
    dedent into stripping the wrong amount from both sides. Simple
    concatenation of two independently-valid, flush-left blocks avoids
    the problem entirely instead of trying to out-clever whitespace math.
    """
    test_runner = (
        "\nimport json as _json\n"
        "_results = []\n"
        f"_tests = {test_cases!r}\n"
        "for _i, _tc in enumerate(_tests):\n"
        "    try:\n"
        f"        _actual = {function_name}(*_tc[\"args\"])\n"
        "        _results.append({\"index\": _i, \"passed\": _actual == _tc[\"expected\"], \"actual\": repr(_actual)})\n"
        "    except Exception as _e:\n"
        "        _results.append({\"index\": _i, \"passed\": False, \"error\": f\"{type(_e).__name__}: {_e}\"})\n"
        "print(\"###RESULTS###\" + _json.dumps(_results))\n"
    )
    return code + "\n\n" + test_runner


def execute_code(code: str, function_name: str, test_cases: list[dict]) -> dict:
    """Runs `code` + a test harness in an isolated subprocess. Returns
    {"passed": bool, "error": str} -- error is "" on success, otherwise
    a message specific enough for the next generate call to act on."""
    harness_src = _build_harness(code, function_name, test_cases)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "attempt.py")
        with open(script_path, "w") as f:
            f.write(harness_src)

        cmd = [sys.executable, "attempt.py"]
        if _HAS_NETNS:
            cmd = ["unshare", "--net"] + cmd

        run_kwargs = {}
        if os.name == "posix" and resource is not None:
            run_kwargs["preexec_fn"] = _limit_resources

        child_env = {"PATH": os.environ.get("PATH", "")}
        for key in ("SystemRoot", "SYSTEMROOT", "SystemDrive", "TEMP", "TMP"):
            if key in os.environ:
                child_env[key] = os.environ[key]

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS,
                env=child_env,  # minimal env: no inherited API keys etc.
                **run_kwargs,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"Timed out after {EXEC_TIMEOUT_SECONDS}s (possible infinite loop)."}

        marker = "###RESULTS###"
        idx = proc.stdout.find(marker)
        if idx == -1:
            if proc.returncode < 0:
                # Negative returncode on POSIX = killed by signal, not a
                # normal Python exception -- this is what an infinite
                # loop or a memory bomb actually looks like: RLIMIT_CPU
                # (5s) or RLIMIT_AS kills it before it ever gets a chance
                # to print anything or raise anything catchable, so
                # stderr is empty. That absence is itself the signal.
                return {
                    "passed": False,
                    "error": (
                        f"Process was killed (signal {-proc.returncode}) before completing -- "
                        "likely an infinite loop, unbounded recursion, or excessive memory use."
                    ),
                }
            # Otherwise: syntax error, missing function, uncaught import
            # error -- something with real stderr to show.
            tail = (proc.stderr or "no stderr captured").strip()[-800:]
            return {"passed": False, "error": f"Code failed to run: {tail}"}

        results = json.loads(proc.stdout[idx + len(marker):])
        failed = [r for r in results if not r["passed"]]
        if not failed:
            return {"passed": True, "error": ""}

        detail = "; ".join(
            f"test {r['index']}: " + (
                r.get("error") or
                f"got {r['actual']}, expected {test_cases[r['index']]['expected']!r}"
            )
            for r in failed
        )
        return {"passed": False, "error": f"{len(failed)}/{len(results)} tests failed -- {detail}"}


# ---------------------------------------------------------------------------
# Generator -- injected, same pattern as lab 4.2's generate_fn. Real
# backend below; smoke_test.py supplies scripted fakes.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:python)?\n|\n```$", re.MULTILINE)


def default_generate(problem: dict, previous_code: Optional[str], error_feedback: Optional[str]) -> str:
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it or put it in a .env file "
            "(this file calls load_dotenv() on import, matching lab 4.2)."
        )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = (
        f"Write a Python function solving this problem:\n{problem['prompt']}\n\n"
        f"The function must be named exactly `{problem['function_name']}`. "
        "Return ONLY the function definition -- no example usage, no explanation, no markdown fences."
    )
    if previous_code and error_feedback:
        prompt += (
            f"\n\nYour previous attempt:\n{previous_code}\n\n"
            f"That attempt failed with: {error_feedback}\n\nFix it."
        )
    raw = llm.invoke(prompt).content.strip()
    # Models routinely add fences despite being told not to -- strip
    # them rather than let a trivial formatting habit break the harness.
    return _FENCE_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_generate_node(generate_fn: Callable[[dict, Optional[str], Optional[str]], str]):
    def generate_node(state: CodeGenState) -> dict:
        iteration = state.get("iteration", 0) + 1
        code = generate_fn(state["problem"], state.get("code"), state.get("last_error"))
        logger.info("problem=%s iteration=%d generated %d chars", state["problem"]["id"], iteration, len(code))
        return {"code": code, "iteration": iteration}

    return generate_node


def execute_node(state: CodeGenState) -> dict:
    problem = state["problem"]
    result = execute_code(state["code"], problem["function_name"], problem["test_cases"])
    logger.info("problem=%s iteration=%d passed=%s", problem["id"], state["iteration"], result["passed"])
    return {"passed": result["passed"], "last_error": result["error"]}


def debug_node(state: CodeGenState) -> dict:
    return {"log": [f"iteration {state['iteration']} failed: {state['last_error']}"]}


def success_node(state: CodeGenState) -> dict:
    return {
        "solved": True,
        "gave_up": False,
        "iterations_used": state["iteration"],
        "log": [f"solved on iteration {state['iteration']}"],
    }


def give_up_node(state: CodeGenState) -> dict:
    return {
        "solved": False,
        "gave_up": True,
        "iterations_used": state["iteration"],
        "log": [f"gave up after {state['iteration']} iterations -- last error: {state['last_error']}"],
    }


def route_after_execute(state: CodeGenState) -> str:
    if state["passed"]:
        return "success"
    if state["iteration"] >= state.get("max_iterations", MAX_ITERATIONS):
        return "give_up"
    return "debug"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph(generate_fn=None):
    generate_fn = generate_fn or default_generate

    builder = StateGraph(CodeGenState)
    builder.add_node("generate", make_generate_node(generate_fn))
    builder.add_node("execute", execute_node)
    builder.add_node("debug", debug_node)
    builder.add_node("success", success_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "execute")
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {"success": "success", "debug": "debug", "give_up": "give_up"},
    )
    builder.add_edge("debug", "generate")
    builder.add_edge("success", END)
    builder.add_edge("give_up", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def solve(problem_id: str):
    problem = PROBLEMS_BY_ID.get(problem_id)
    if not problem:
        console.print(f"[red]Unknown problem_id '{problem_id}'. Choices: {', '.join(PROBLEMS_BY_ID)}[/red]")
        raise typer.Exit(1)

    graph = build_graph()
    try:
        result = graph.invoke({"problem": problem, "max_iterations": MAX_ITERATIONS})
    except RuntimeError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

    style = "green" if result["solved"] else "red"
    title = f"{'solved' if result['solved'] else 'gave up'} after {result['iterations_used']} iteration(s)"
    console.print(Panel(result["code"], title=title, style=style))
    for line in result["log"]:
        console.print(f"  [dim]{line}[/dim]")


@app.command(name="run-all")
def run_all():
    """Run every problem in the suite, print a summary table and the overall solve rate."""
    graph = build_graph()
    table = Table(title="Iterative Code Generator -- Full Suite")
    table.add_column("Problem")
    table.add_column("Result")
    table.add_column("Iterations")

    solved_count = 0
    for problem in PROBLEMS:
        try:
            result = graph.invoke({"problem": problem, "max_iterations": MAX_ITERATIONS})
        except RuntimeError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1)
        solved_count += int(result["solved"])
        table.add_row(
            problem["id"],
            "[green]solved[/green]" if result["solved"] else "[red]gave up[/red]",
            str(result["iterations_used"]),
        )
    console.print(table)

    rate = solved_count / len(PROBLEMS) * 100
    style = "green" if rate >= 70 else "red"
    console.print(f"[{style}]Solve rate: {solved_count}/{len(PROBLEMS)} ({rate:.0f}%)[/{style}]")


if __name__ == "__main__":
    app()
