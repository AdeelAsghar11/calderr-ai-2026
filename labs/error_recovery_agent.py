import os
import time
import random
import logging
from datetime import datetime
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.rule import Rule
# pyrefly: ignore [missing-import]
from rich.table import Table

load_dotenv()

console = Console()

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    filename="agent_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── Tools — some deliberately fail ───────────────────────────────
@tool
def primary_search(query: str) -> str:
    """Search the primary database for information. May occasionally fail."""
    # Simulate 40% failure rate
    if random.random() < 0.4:
        raise ConnectionError(f"Primary database unavailable for query: {query}")
    return f"Primary DB result for '{query}': Found relevant data about {query}."


@tool
def backup_search(query: str) -> str:
    """Search the backup database. Use when primary_search fails."""
    # Simulate 10% failure rate
    if random.random() < 0.1:
        raise ConnectionError(f"Backup database also unavailable for query: {query}")
    return f"Backup DB result for '{query}': Found cached data about {query}."


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        import math
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    except Exception as e:
        raise ValueError(f"Invalid expression '{expression}': {e}")


@tool
def unreliable_api(endpoint: str) -> str:
    """Call an external API endpoint. This tool fails 60% of the time."""
    if random.random() < 0.6:
        raise TimeoutError(f"API timeout for endpoint: {endpoint}")
    return f"API response from {endpoint}: success"


@tool
def static_fallback(query: str = "", endpoint: str = "") -> str:
    """Last resort fallback. Always succeeds. Use when all other tools fail."""
    topic = query or endpoint or "your request"
    return f"Fallback response for '{topic}': Unable to fetch live data at this time."


# ── Tool registry with fallback mapping ──────────────────────────
ALL_TOOLS = [primary_search, backup_search, calculate, unreliable_api, static_fallback]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

FALLBACK_MAP = {
    "primary_search": ["backup_search", "static_fallback"],
    "unreliable_api": ["static_fallback"],
    "backup_search":  ["static_fallback"],
}


# ── Exponential backoff ───────────────────────────────────────────
def with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = (base_delay * (2 ** attempt)) + random.random()
                console.print(f"    [yellow]⚠ Attempt {attempt + 1} failed: {type(e).__name__}[/yellow]")
                console.print(f"    [dim]  Retrying in {wait:.1f}s...[/dim]")
                log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
            else:
                console.print(f"    [red]✗ All {max_retries} attempts failed.[/red]")
                log.error(f"All {max_retries} attempts failed. Last error: {e}")
    if last_error is None:
        raise ValueError("max_retries must be at least 1")
    raise last_error


# ── Tool executor with fallback logic ────────────────────────────
def execute_tool_with_recovery(tool_name: str, tool_args: dict) -> tuple[str, str]:
    """
    Try the requested tool, then fallbacks if it fails.
    Returns (result, tool_actually_used)
    """
    tools_to_try = [tool_name] + FALLBACK_MAP.get(tool_name, [])

    for current_tool in tools_to_try:
        console.print(f"  [cyan]→ Trying:[/cyan] [bold]{current_tool}[/bold]  args={tool_args}")
        log.info(f"Attempting tool: {current_tool} with args: {tool_args}")

        try:
            result = with_backoff(
                lambda t=current_tool, a=tool_args: TOOL_MAP[t].invoke(a),
                max_retries=2
            )
            if current_tool != tool_name:
                console.print(f"  [green]✓ Recovered using fallback:[/green] {current_tool}")
                log.info(f"Recovered using fallback: {current_tool}")
            else:
                console.print(f"  [green]✓ Success[/green]")
                log.info(f"Tool succeeded: {current_tool}")
            return result, current_tool

        except Exception as e:
            console.print(f"  [red]✗ {current_tool} failed: {type(e).__name__}[/red]")
            log.error(f"Tool {current_tool} failed: {e}")
            if current_tool == tools_to_try[-1]:
                # All fallbacks exhausted
                error_msg = f"All tools failed for this request. Last error: {e}"
                log.critical(f"All fallbacks exhausted for {tool_name}")
                return error_msg, "none"

    return "Unexpected error", "none"


# ── Agent loop ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research assistant with these tools:
- primary_search: search the primary database (may fail)
- backup_search: search backup database (use if primary fails)  
- calculate: evaluate math expressions
- unreliable_api: call an external API (often fails)
- static_fallback: always works, use as last resort

Rules:
- Call ONLY the tools needed to answer the question. Do not call extra tools.
- After getting tool results, give your final answer immediately.
- For a search question: call ONE search tool, then answer.
- For a math question: call calculate once, then answer.
- For an API question: call unreliable_api once, then answer.
- Never call more than 3 tools per question."""


def run_agent(question: str, llm) -> dict:
    console.print(f"\n[bold white]Question:[/bold white] {question}")
    console.print(Rule(style="dim"))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    attempts_log = []
    step = 0

    while step < 8:
        step += 1
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            console.print(Panel(
                response.content,
                title="[green]Final Answer[/green]",
                border_style="green"
            ))
            log.info(f"Final answer given for: {question}")
            return {"answer": response.content, "attempts": attempts_log}

        for tc in response.tool_calls:
            result, tool_used = execute_tool_with_recovery(tc["name"], tc["args"])
            attempts_log.append({
                "requested": tc["name"],
                "used": tool_used,
                "success": tool_used != "none"
            })
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            ))

    return {"answer": "Max steps reached", "attempts": attempts_log}


def main() -> None:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    ).bind_tools(ALL_TOOLS)

    console.print(Rule("[bold cyan]Lab 2.3 — Error Recovery Agent[/bold cyan]"))
    console.print("[dim]Tools have random failure rates. Watch the recovery logic.[/dim]\n")

    questions = [
        "Search for information about machine learning.",
        "What is 128 * 256?",
        "Call the weather API endpoint '/weather/karachi'.",
        "Search for Python programming news.",
        "Call the API at '/stock/AAPL' and search for stock market news.",
    ]

    all_attempts = []

    for q in questions:
        result = run_agent(q, llm)
        all_attempts.extend(result["attempts"])
        console.print()

    # ── Summary table ─────────────────────────────────────────────
    console.print(Rule("[bold cyan]Recovery Summary[/bold cyan]"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Requested Tool",  width=20)
    table.add_column("Actually Used",   width=20)
    table.add_column("Recovered",       width=10)

    recovered = 0
    for a in all_attempts:
        was_recovered = a["requested"] != a["used"] and a["success"]
        if was_recovered:
            recovered += 1
        table.add_row(
            a["requested"],
            a["used"],
            "[green]✓ Yes[/green]" if was_recovered else
            "[red]✗ Failed[/red]" if not a["success"] else
            "[dim]—[/dim]"
        )

    console.print(table)
    console.print(f"\n[green]Recovered from {recovered} tool failures[/green]")
    console.print("[dim]Full log saved to agent_log.txt[/dim]")


if __name__ == "__main__":
    main()