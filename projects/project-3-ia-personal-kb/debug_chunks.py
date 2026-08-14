"""
debug_chunks.py — Inspect Raw Ingested Chunks

Diagnostic tool: prints every stored chunk (optionally from one source)
whose raw text contains a keyword. Use this to check whether text actually
made it into the knowledge base intact, or got scrambled/dropped during
PDF extraction — a different failure mode than retrieval ranking.

Usage
-----
  python debug_chunks.py faithfulness
  python debug_chunks.py "RAGAS metrics" --source calderr_week_3
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel

# pyrefly: ignore [missing-import]
from ingest import get_collection

app     = typer.Typer(add_completion=False)
console = Console()


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Text to search for in raw stored chunks"),
    source:  str = typer.Option(None, "--source", "-s", help="Limit to one document slug"),
) -> None:
    """Search raw chunk text directly in ChromaDB, bypassing retrieval ranking entirely."""
    collection = get_collection()
    where = {"source": {"$eq": source}} if source else None
    data  = collection.get(where=where, include=["documents", "metadatas"])

    matches = 0
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if keyword.lower() in doc.lower():
            matches += 1
            console.print(Panel(
                doc,
                title=f"{meta.get('filename', '?')}  chunk {meta.get('chunk_index', '?')}",
                border_style="green",
            ))

    if matches == 0:
        console.print(f"\n[red]No chunk contains '{keyword}'"
                       + (f" in {source}" if source else "") + ".[/red]")
        console.print("[yellow]This usually means the source text was scrambled or split "
                       "during PDF extraction — not that retrieval ranked it poorly.[/yellow]\n")
    else:
        console.print(f"\n[green]Found in {matches} chunk(s).[/green]\n")


if __name__ == "__main__":
    app()