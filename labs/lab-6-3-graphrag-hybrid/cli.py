"""
cli.py — Typer + Rich CLI interface for Lab 6.3 GraphRAG Hybrid Retrieval.

Commands:
1. run-all: Executes the benchmark suite over all 15 questions and prints a rich comparison table.
2. query: Runs an ad-hoc natural language question through the pipeline and prints router decision + final answer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Ensure package directory is in sys.path
LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

try:
    # pyrefly: ignore [missing-import]
    from .hybrid_retriever import GraphRAGHybridRetriever
    # pyrefly: ignore [missing-import]
    from .models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from .router import QueryRouter
    # pyrefly: ignore [missing-import]
    from .smoke_test import BENCHMARK_QUESTIONS
except ImportError:
    # pyrefly: ignore [missing-import]
    from hybrid_retriever import GraphRAGHybridRetriever
    # pyrefly: ignore [missing-import]
    from models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from router import QueryRouter
    # pyrefly: ignore [missing-import]
    from smoke_test import BENCHMARK_QUESTIONS

app = typer.Typer(help="Lab 6.3 GraphRAG Hybrid Retrieval CLI")
console = Console()


@app.command(name="run-all")
def run_all(
    real: bool = typer.Option(
        False, "--real", help="Run router and answer generation in real LLM mode using ChatGroq."
    )
) -> None:
    """
    Run all 15 benchmark questions through the full GraphRAG hybrid pipeline.
    Displays a structured comparison table showing True Category, Predicted Category, Method Used, and Correctness.
    """
    console.print(
        f"\n[bold green]Running GraphRAG Hybrid Retrieval Benchmark (Mode: {'--real' if real else 'stub'})[/bold green]\n"
    )

    retriever = GraphRAGHybridRetriever(use_real=real)

    table = Table(
        title="GraphRAG Hybrid Retrieval - 15 Question Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", width=45)
    table.add_column("True Cat", style="cyan", width=10)
    table.add_column("Pred Cat", style="yellow", width=10)
    table.add_column("Method Used", style="blue", width=12)
    table.add_column("Correct", style="bold green", width=8)

    correct_count = 0

    for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
        # Route query to select method
        router_decision = retriever.router.route(q)
        # Execute pipeline with router-selected method
        res = retriever.process_question(q)

        if res.is_correct:
            correct_count += 1
            correct_str = "[green]PASS[/green]"
        else:
            correct_str = "[red]FAIL[/red]"

        table.add_row(
            str(i),
            q.question,
            q.category,
            router_decision.predicted_category,
            res.method,
            correct_str,
        )

    console.print(table)
    console.print(
        f"\n[bold yellow]Benchmark Summary:[/bold yellow] Correct Retrieval Context: [bold green]{correct_count}/15[/bold green] ({(correct_count/15)*100:.1f}%)\n"
    )


@app.command(name="query")
def query_command(
    question: str = typer.Argument(..., help="Natural language question to query."),
    real: bool = typer.Option(
        False, "--real", help="Run router and answer generation in real LLM mode using ChatGroq."
    ),
    method: Optional[str] = typer.Option(
        None, "--method", help="Override router and force a specific method: vector_only, graph_only, or hybrid."
    ),
) -> None:
    """
    Query the GraphRAG hybrid retrieval system with an ad-hoc question.
    """
    console.print(f"\n[bold cyan]Querying GraphRAG Hybrid System:[/bold cyan] '{question}'\n")

    retriever = GraphRAGHybridRetriever(use_real=real)
    router = QueryRouter(use_real=real)

    predicted_cat = router.classify(question)
    console.print(f"[bold yellow]Router Decision:[/bold yellow] Predicted Category = [bold green]{predicted_cat}[/bold green]")

    dummy_record = QuestionRecord(
        question=question,
        category=predicted_cat,
        expected_answer_keywords=[],
    )

    override_m = method if method in ("vector_only", "graph_only", "hybrid") else None
    res = retriever.process_question(dummy_record, override_method=override_m)  # type: ignore[arg-type]

    console.print(f"[bold yellow]Method Used:[/bold yellow] {res.method}")
    console.print(f"\n[bold yellow]Retrieved Context Used:[/bold yellow]\n{res.context_used}")
    console.print(f"\n[bold green]Final Generated Answer:[/bold green]\n{res.answer}\n")


if __name__ == "__main__":
    app()
