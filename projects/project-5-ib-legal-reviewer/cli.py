"""
Typer CLI interface for Project 5-I-B: Multi-Agent Legal Document Reviewer.
Renders the debate cross-examination transcript and final legal report using Rich tables and panels.
"""

import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add project root to sys.path
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from agents import run_legal_review
except ImportError:
    from project_5_ib_legal_reviewer.agents import run_legal_review

app = typer.Typer(
    help="Multi-Agent Legal Document Reviewer CLI",
    add_completion=False,
)
console = Console()


def get_severity_style(severity: int) -> str:
    """Color styling helper based on severity level."""
    if severity >= 5:
        return "bold red"
    elif severity == 4:
        return "bold orange3"
    elif severity == 3:
        return "bold yellow"
    elif severity == 2:
        return "bold green"
    return "dim green"


@app.command()
def review(
    contract_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the contract document text file to review",
    ),
    real: bool = typer.Option(
        False,
        "--real",
        help="Use real ChatGroq LLM mode (requires GROQ_API_KEY) instead of offline stub mode",
    ),
):
    """
    Run multi-agent legal document review on a target contract file.
    """
    document_name = contract_path.name
    console.print(
        f"\n[bold blue]Legal Document Reviewer[/bold blue] - Analyzing document: [bold yellow]{document_name}[/bold yellow] (Mode: {'REAL (Groq Llama-3.3-70B)' if real else 'STUB (Offline Deterministic)'})\n"
    )

    with open(contract_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    try:
        report = run_legal_review(document_name, document_text, real_mode=real)
    except Exception as e:
        console.print(f"[bold red]Error during legal review:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 1. Debate Cross-Examination Transcript Table
    console.print("[bold cyan]Debate Cross-Examination Transcript[/bold cyan]")
    transcript_table = Table(title="Peer Specialist Cross-Examination", show_header=True, header_style="bold magenta")
    transcript_table.add_column("Challenger", style="cyan", width=18)
    transcript_table.add_column("Target Clause", style="white", width=25)
    transcript_table.add_column("Stance", width=10)
    transcript_table.add_column("Legal Reasoning", style="dim white")

    for challenge in report.debate_transcript:
        stance_style = "bold red" if challenge.stance == "dispute" else "bold green"
        transcript_table.add_row(
            challenge.challenger,
            challenge.target_clause_reference,
            Text(challenge.stance.upper(), style=stance_style),
            challenge.reasoning,
        )

    console.print(transcript_table)
    console.print()

    # 2. Final Synthesized Findings Table
    console.print("[bold cyan]Final Synthesized Clause Findings (Judge Agent Output)[/bold cyan]")
    findings_table = Table(title=f"Review Report: {document_name}", show_header=True, header_style="bold blue")
    findings_table.add_column("Clause Reference", style="white", width=25)
    findings_table.add_column("Raised By", style="cyan", width=18)
    findings_table.add_column("Concern / Risk", style="white", width=40)
    findings_table.add_column("Final Severity", width=14, justify="center")
    findings_table.add_column("Contested?", width=12, justify="center")
    findings_table.add_column("Dissent Notes", style="dim red")

    for finding in report.findings:
        sev_style = get_severity_style(finding.final_severity)
        contested_text = Text("YES", style="bold red") if finding.contested else Text("NO", style="dim green")
        dissent_str = "\n".join(finding.dissent_notes) if finding.dissent_notes else "-"

        findings_table.add_row(
            finding.clause_reference,
            finding.raised_by,
            finding.concern,
            Text(f"{finding.final_severity} / 5", style=sev_style),
            contested_text,
            dissent_str,
        )

    console.print(findings_table)
    console.print()

    # 3. Executive Risk Summary Panel
    summary_panel = Panel(
        report.overall_risk_summary,
        title="[bold yellow]Executive Risk Summary[/bold yellow]",
        border_style="yellow",
        expand=True,
    )
    console.print(summary_panel)
    console.print("\n[bold green]Review complete.[/bold green]\n")


if __name__ == "__main__":
    app()
