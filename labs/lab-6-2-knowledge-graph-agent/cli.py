"""
cli.py — Typer + Rich CLI interface for Lab 6.2 Knowledge Graph Agent.

Commands:
- query <question>: Answers a multi-hop question, rendering the graph traversal path as a Table and answer as a Panel.
- build: Rebuilds the knowledge graph from sample_corpus.py and renders the interactive Pyvis HTML file.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    # pyrefly: ignore [missing-import]
    from .graph_builder import build_knowledge_graph, render_pyvis_graph
    # pyrefly: ignore [missing-import]
    from .query_agent import KnowledgeGraphQueryAgent
    # pyrefly: ignore [missing-import]
    from .sample_corpus import CORPUS_PARAGRAPHS
except ImportError:
    # pyrefly: ignore [missing-import]
    from graph_builder import build_knowledge_graph, render_pyvis_graph
    # pyrefly: ignore [missing-import]
    from query_agent import KnowledgeGraphQueryAgent
    # pyrefly: ignore [missing-import]
    from sample_corpus import CORPUS_PARAGRAPHS

app = typer.Typer(help="Lab 6.2 Knowledge Graph Agent CLI")
console = Console()


@app.command()
def query(
    question: str = typer.Argument(..., help="The natural language question to ask the knowledge graph agent"),
    real: bool = typer.Option(False, "--real", help="Use real Groq LLM API instead of offline stub"),
    html_out: str = typer.Option("graph.html", "--html-out", help="Path to write Pyvis HTML visualization"),
) -> None:
    """
    Query the knowledge graph agent with a multi-hop question.
    """
    console.print(Panel(f"[bold cyan]Question:[/bold cyan] {question}", title="Lab 6.2 Knowledge Graph Agent"))

    # Build knowledge graph
    graph = build_knowledge_graph(CORPUS_PARAGRAPHS, use_real=real)
    render_pyvis_graph(graph, output_file=html_out)

    # Initialize query agent and run query
    agent = KnowledgeGraphQueryAgent(graph=graph, corpus=CORPUS_PARAGRAPHS, use_real=real)
    result = agent.answer_query(question)

    # Render traversal hops in a Rich Table
    table = Table(title="Graph Traversal Path (Reasoning Trace)", header_style="bold magenta")
    table.add_column("From Entity", style="cyan")
    table.add_column("Relationship", style="yellow")
    table.add_column("To Entity", style="green")
    table.add_column("Direction", style="dim white")

    for hop in result.path:
        table.add_row(hop.from_entity, hop.relationship, hop.to_entity, hop.direction)

    console.print(table)

    # Render Answer Panel
    kw_status = (
        "[bold green]FAILED[/bold green] (Expected: Facts are in separate paragraphs; Graph Traversal was required & succeeded!)"
        if not result.keyword_search_would_succeed
        else "[bold yellow]SUCCEEDED[/bold yellow] (Single paragraph contained both entities)"
    )

    console.print(
        Panel(
            f"[bold green]Answer:[/bold green] {result.answer}\n"
            f"Grounded Entity: [cyan]{result.grounded_entities}[/cyan]\n"
            f"Keyword Search Baseline: {kw_status}",
            title="Query Result",
            border_style="green",
        )
    )


@app.command()
def build(
    html_out: str = typer.Option("graph.html", "--html-out", help="Path to save Pyvis HTML visualization"),
    real: bool = typer.Option(False, "--real", help="Use real Groq LLM API instead of offline stub"),
) -> None:
    """
    Rebuild the knowledge graph from sample_corpus.py and generate Pyvis HTML visualization.
    """
    console.print("[yellow]Building Knowledge Graph from sample corpus...[/yellow]")
    graph = build_knowledge_graph(CORPUS_PARAGRAPHS, use_real=real)
    out_path = render_pyvis_graph(graph, output_file=html_out)

    console.print(
        Panel(
            f"Knowledge Graph successfully built!\n"
            f"Nodes: [cyan]{graph.number_of_nodes()}[/cyan]\n"
            f"Edges: [cyan]{graph.number_of_edges()}[/cyan]\n"
            f"HTML Visualization saved to: [green]{out_path.resolve()}[/green]",
            title="Graph Build Summary",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    app()
