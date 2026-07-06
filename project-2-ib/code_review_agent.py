import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.syntax import Syntax

load_dotenv()

console = Console()


# ── Pydantic models ───────────────────────────────────────────────
class Issue(BaseModel):
    line: Optional[int] = Field(None, description="Line number where the issue occurs, null if general")
    description: str = Field(description="Clear description of the issue")
    suggestion: str = Field(description="How to fix it")
    severity: str = Field(description="One of: critical, high, medium, low")


class ReviewPass(BaseModel):
    category: str = Field(description="Review category: bugs, security, style, or performance")
    issues: list[Issue] = Field(description="List of issues found, empty list if none")
    summary: str = Field(description="One sentence summary of findings for this category")
    score: int = Field(description="Code quality score for this category, 1-10", ge=1, le=10)


class CodeReviewReport(BaseModel):
    overall_score: int = Field(description="Overall code quality score 1-10", ge=1, le=10)
    verdict: str = Field(description="One of: approve, approve_with_changes, request_changes, reject")
    passes: list[ReviewPass] = Field(description="Results from all 4 review passes")
    top_priority: str = Field(description="The single most important thing to fix")


# ── Review prompts per category ───────────────────────────────────
REVIEW_PROMPTS = {
    "bugs": """You are a senior Python engineer reviewing code for bugs and correctness issues.
Look for: logic errors, off-by-one errors, unhandled exceptions, incorrect variable usage,
infinite loops, wrong return values, missing null checks, type errors.
Be thorough but only report real issues, not style preferences.""",

    "security": """You are a security engineer reviewing Python code for vulnerabilities.
Look for: SQL injection, command injection, eval/exec with user input, hardcoded secrets,
insecure random number generation, path traversal, missing input validation,
exposed sensitive data, insecure deserialization, OWASP Top 10 issues.""",

    "style": """You are a Python style reviewer enforcing PEP 8 and clean code principles.
Look for: poor naming conventions, missing docstrings, overly long functions,
magic numbers, commented-out code, inconsistent formatting, poor variable names,
missing type hints, overly complex conditions.""",

    "performance": """You are a performance engineer reviewing Python code for inefficiencies.
Look for: O(n²) where O(n) is possible, repeated computations in loops,
unnecessary list comprehensions, missing caching, inefficient string concatenation,
loading entire files into memory, N+1 query patterns, missing generators.""",
}

REVIEW_TEMPLATE = """{category_prompt}

Review the following Python code and return structured feedback.

```python
{code}
```

Return a ReviewPass object for the '{category}' category."""


# ── Agent functions ───────────────────────────────────────────────
def run_review_pass(code: str, category: str, llm) -> ReviewPass:
    structured_llm = llm.with_structured_output(ReviewPass)
    prompt = ChatPromptTemplate.from_messages([
        ("user", REVIEW_TEMPLATE)
    ])
    chain = prompt | structured_llm
    return chain.invoke({
        "category_prompt": REVIEW_PROMPTS[category],
        "code": code,
        "category": category,
    })


def synthesize_report(code: str, passes: list[ReviewPass], llm) -> CodeReviewReport:
    structured_llm = llm.with_structured_output(CodeReviewReport)

    passes_text = "\n\n".join([
        f"Category: {p.category} | Score: {p.score}/10\n"
        f"Summary: {p.summary}\n"
        f"Issues: {len(p.issues)} found\n" +
        "\n".join(f"  - [{i.severity.upper()}] Line {i.line or 'N/A'}: {i.description}" for i in p.issues)
        for p in passes
    ])

    # Escape curly braces so LangChain doesn't treat them as template variables
    safe_code = code.replace("{", "{{").replace("}", "}}")
    safe_passes = passes_text.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a lead engineer synthesizing a code review. Be fair but direct."),
        ("user",
         f"Code reviewed:\n```python\n{safe_code}\n```\n\n"
         f"Review findings:\n{safe_passes}\n\n"
         "Produce a final CodeReviewReport with overall score, verdict, and top priority fix.")
    ])
    chain = prompt | structured_llm
    return chain.invoke({})


# ── Display functions ─────────────────────────────────────────────
SEVERITY_COLORS = {
    "critical": "bold red",
    "high":     "red",
    "medium":   "yellow",
    "low":      "dim",
}

VERDICT_COLORS = {
    "approve":               "bold green",
    "approve_with_changes":  "green",
    "request_changes":       "yellow",
    "reject":                "bold red",
}

SCORE_COLORS = {
    range(1, 4):  "red",
    range(4, 7):  "yellow",
    range(7, 11): "green",
}


def get_score_color(score: int) -> str:
    for r, color in SCORE_COLORS.items():
        if score in r:
            return color
    return "white"


def display_report(report: CodeReviewReport, passes: list[ReviewPass]) -> None:
    # ── Overall verdict ───────────────────────────────────────────
    verdict_color = VERDICT_COLORS.get(report.verdict, "white")
    console.print(Panel(
        f"[{verdict_color}]Verdict: {report.verdict.upper().replace('_', ' ')}[/{verdict_color}]\n"
        f"[{get_score_color(report.overall_score)}]Overall Score: {report.overall_score}/10[/{get_score_color(report.overall_score)}]\n\n"
        f"[bold]Top Priority:[/bold] {report.top_priority}",
        title="[bold cyan]Code Review Summary[/bold cyan]",
        border_style="cyan"
    ))

    # ── Per-category results ──────────────────────────────────────
    for p in passes:
        color = get_score_color(p.score)
        console.print(f"\n[bold]{p.category.upper()}[/bold] — Score: [{color}]{p.score}/10[/{color}] — {p.summary}")

        if not p.issues:
            console.print("  [green]✓ No issues found[/green]")
            continue

        table = Table(show_header=True, header_style="bold", show_lines=True)
        table.add_column("Severity", width=10)
        table.add_column("Line",     width=5)
        table.add_column("Issue",    width=40)
        table.add_column("Fix",      width=40)

        for issue in sorted(p.issues, key=lambda x: ["critical","high","medium","low"].index(x.severity.lower())):
            sev_color = SEVERITY_COLORS.get(issue.severity.lower(), "white")
            table.add_row(
                f"[{sev_color}]{issue.severity.upper()}[/{sev_color}]",
                str(issue.line or "—"),
                issue.description,
                issue.suggestion,
            )
        console.print(table)

    # ── Category scores summary ───────────────────────────────────
    console.print(Rule("[bold cyan]Category Scores[/bold cyan]"))
    score_table = Table(show_header=True, header_style="bold cyan")
    score_table.add_column("Category",   width=15)
    score_table.add_column("Score",      width=8)
    score_table.add_column("Issues",     width=8)
    score_table.add_column("Summary",    width=50)

    for p in passes:
        color = get_score_color(p.score)
        score_table.add_row(
            p.category.capitalize(),
            f"[{color}]{p.score}/10[/{color}]",
            str(len(p.issues)),
            p.summary[:48],
        )
    console.print(score_table)


# ── Sample code snippets to review ───────────────────────────────
SAMPLE_CODES = {
    "buggy_auth": '''
def authenticate_user(username, password):
    users = {"admin": "password123", "user": "abc123"}
    if username in users:
        if users[username] == password:
            return True
    return False

def get_user_data(user_id):
    import sqlite3
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    return cursor.fetchall()

def process_file(filename):
    data = open(filename).read()
    result = eval(data)
    return result
''',

    "performance_issues": '''
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

def load_and_process(filename):
    with open(filename) as f:
        lines = f.readlines()
    result = ""
    for line in lines:
        result = result + line.strip() + " "
    return result

def get_user_names(user_ids):
    names = []
    for uid in user_ids:
        user = database.query(f"SELECT name FROM users WHERE id={uid}")
        names.append(user.name)
    return names
''',
}


def review_code(code: str, label: str = "Code") -> None:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    console.print(Rule(f"[bold cyan]Reviewing: {label}[/bold cyan]"))
    console.print(Syntax(code.strip(), "python", theme="monokai", line_numbers=True))
    console.print()

    passes = []
    for category in ["bugs", "security", "style", "performance"]:
        with console.status(f"[dim]Running {category} pass...[/dim]", spinner="dots"):
            result = run_review_pass(code, category, llm)
            passes.append(result)
        console.print(f"  [green]✓[/green] {category}: {len(result.issues)} issues found")

    with console.status("[dim]Synthesizing final report...[/dim]", spinner="dots"):
        report = synthesize_report(code, passes, llm)

    display_report(report, passes)


def main() -> None:
    console.print(Rule("[bold cyan]Project 2-I-B — Code Review Agent[/bold cyan]"))
    console.print("[dim]4-pass review: bugs · security · style · performance[/dim]\n")

    for label, code in SAMPLE_CODES.items():
        review_code(code, label)
        console.print("\n")


if __name__ == "__main__":
    main()