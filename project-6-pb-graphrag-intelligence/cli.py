"""
cli.py — Typer + Rich CLI interface for Project 6-PB Phase 3 GraphRAG System.

Commands:
1. run-study: Execute 30-question study evaluation and paired t-test analysis.
2. serve-api: Launch FastAPI control plane on port 8000.
3. serve-dashboard: Launch Streamlit research dashboard on port 8501.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJ_DIR = Path(__file__).resolve().parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# Load environment variables from .env in repository root
load_dotenv()

try:
    # pyrefly: ignore [missing-import]
    from .dataset import get_verified_benchmark_dataset
    from .evaluator import EvaluationRunner
    from .report_generator import generate_html_report
    from .statistical_test import run_paired_ttest
except ImportError:
    from dataset import get_verified_benchmark_dataset
    from evaluator import EvaluationRunner
    from report_generator import generate_html_report
    from statistical_test import run_paired_ttest

app = typer.Typer(help="Project 6-PB GraphRAG Phase 3 System CLI")
console = Console()


@app.command(name="run-study")
def run_study(
    real: bool = typer.Option(
        False, "--real", help="Run evaluation using real ChatGroq LLM RAGAS scoring."
    )
) -> None:
    """
    Execute 30-question evaluation across vector_only, graph_only, and hybrid methods.
    Calculates paired t-test statistical significance and exports evaluation_report.html.
    """
    mode_str = "REAL RAGAS MODE (ChatGroq llama-3.3-70b-versatile)" if real else "STUB MODE (Plumbing Verification)"
    console.print(f"\n[bold green]Running Project 6-PB GraphRAG Evaluation Study ({mode_str})[/bold green]\n")

    dataset = get_verified_benchmark_dataset()
    console.print(f"[bold cyan]Verified Benchmark Dataset:[/bold cyan] {len(dataset)} questions (10 Factual, 10 Relational, 10 Complex)\n")

    runner = EvaluationRunner(use_real=real)

    with console.status(f"[bold yellow]Executing 90 evaluation runs ({mode_str})...[/bold yellow]"):
        records = runner.run_evaluation(dataset)

    console.print(f"[bold green]Successfully collected {len(records)} evaluation records (30 questions x 3 methods).[/bold green]\n")

    # Calculate category x method summary table
    categories = ["factual", "relational", "complex"]
    methods = ["vector_only", "graph_only", "hybrid"]

    table = Table(title="GraphRAG 30-Question Evaluation Summary (Mean RAGAS Score)", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Vector-Only", style="yellow")
    table.add_column("Graph-Only", style="blue")
    table.add_column("Hybrid", style="bold green")

    for cat in categories:
        row_vals = [cat.upper()]
        for m in methods:
            cat_m_recs = [r for r in records if r.category == cat and r.method == m]
            if cat_m_recs:
                avg = sum((r.faithfulness + r.response_relevancy + r.context_precision + r.context_recall) / 4.0 for r in cat_m_recs) / len(cat_m_recs)
                row_vals.append(f"{avg:.3f}")
            else:
                row_vals.append("N/A")
        table.add_row(*row_vals)

    console.print(table)
    console.print()

    # Calculate Paired t-test statistical significance on complex category
    sig_result = run_paired_ttest(records, category="complex")

    sig_text = (
        f"[bold white]Paired t-Test Results (Complex Category - Hybrid vs Vector-Only):[/bold white]\n\n"
        f"  - [bold yellow]Sample Size (n):[/bold yellow] {sig_result.sample_size}\n"
        f"  - [bold yellow]t-Statistic:[/bold yellow] {sig_result.t_statistic:.4f}\n"
        f"  - [bold yellow]p-Value:[/bold yellow] {sig_result.p_value:.6f}\n"
        f"  - [bold yellow]Statistically Significant (p < 0.05):[/bold yellow] "
        f"{'[bold green]YES (True)[/bold green]' if sig_result.significant_at_05 else '[bold red]NO (False)[/bold red]'}\n\n"
        f"[dim]Note: This evaluation was executed in {'REAL RAGAS mode' if real else 'STUB mode'}.[/dim]"
    )

    console.print(Panel(sig_text, title="[bold green]Statistical Significance Panel[/bold green]", border_style="green"))

    # Export HTML Report
    html_path = generate_html_report(records, sig_result, is_real=real)
    console.print(f"\n[bold green]Exported HTML Evaluation Report to:[/bold green] [cyan]{html_path}[/cyan]\n")


@app.command(name="serve-api")
def serve_api(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address for FastAPI server."),
    port: int = typer.Option(8000, "--port", help="Port for FastAPI server."),
) -> None:
    """Launch FastAPI control plane using uvicorn."""
    import uvicorn
    console.print(f"\n[bold green]Starting FastAPI Control Plane on http://{host}:{port}[/bold green]\n")
    uvicorn.run("project-6-pb-graphrag-intelligence.api:app", host=host, port=port, reload=False)


@app.command(name="serve-dashboard")
def serve_dashboard(
    port: int = typer.Option(8501, "--port", help="Port for Streamlit dashboard."),
) -> None:
    """Launch Streamlit research dashboard."""
    import subprocess
    dash_path = PROJ_DIR / "dashboard.py"
    console.print(f"\n[bold green]Starting Streamlit Research Dashboard on http://localhost:{port}[/bold green]\n")
    subprocess.run(["uv", "run", "streamlit", "run", str(dash_path), "--server.port", str(port)])


if __name__ == "__main__":
    app()
