"""
query.py — Step 3: Query the ChromaDB collection

Demonstrates the core vector DB query patterns used in RAG:
  1. Plain semantic search
  2. Filter by source document
  3. Filter by metadata (word_count, date)
  4. Combination filters

ChromaDB uses MongoDB-style where clauses for metadata filtering.
These are applied BEFORE the vector search, so they reduce the candidate set.

Commands
--------
  python query.py ask "what is a transformer architecture?"
  python query.py ask "how does learning work?" --source deep_learning
  python query.py ask "retrieval augmented generation" --top-k 8
  python query.py ask "language models" --min-words 60 --source large_language_model
  python query.py sources         # list available source documents
  python query.py peek ml         # show first 5 chunks of a document
"""

from __future__ import annotations

import textwrap

import chromadb
import typer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingest import CHROMA_PATH, EMBEDDING_MODEL, collection_name

app     = typer.Typer(help="ChromaDB query CLI — Lab 3.2", add_completion=False)
console = Console()

SNIPPET_LEN = 300   # characters to show per result chunk


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_collection(chunk_size: int) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    cname  = collection_name(chunk_size)
    try:
        return client.get_collection(name=cname, embedding_function=ef)
    except Exception:
        console.print(f"[red]Collection '{cname}' not found. Run ingest.py first.[/red]")
        raise typer.Exit(1)


def score_colour(distance: float) -> str:
    """
    ChromaDB returns DISTANCE (lower = more similar) when using cosine space.
    Convert to a rough quality label.
      distance ~0.0  → near-identical
      distance ~0.5  → moderately related
      distance ~1.0  → unrelated
    """
    if distance < 0.25:   return "bright_green"
    if distance < 0.45:   return "green"
    if distance < 0.65:   return "yellow"
    return "red"


def build_where(source: str | None, min_words: int | None) -> dict | None:
    """
    Build a ChromaDB metadata filter.

    ChromaDB uses MongoDB-style operators:
      $eq, $ne, $gt, $gte, $lt, $lte  — for single field conditions
      $and, $or                         — to combine conditions
    """
    conditions = []
    if source:
        conditions.append({"source": {"$eq": source}})
    if min_words:
        conditions.append({"word_count": {"$gte": min_words}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def display_results(results: dict, query: str) -> None:
    """Render query results as a rich table."""
    docs       = results.get("documents", [[]])[0]
    metadatas  = results.get("metadatas",  [[]])[0]
    distances  = results.get("distances",  [[]])[0]
    ids        = results.get("ids",        [[]])[0]

    if not docs:
        console.print("[yellow]No results found for this query + filter combination.[/yellow]")
        return

    console.print()
    t = Table(
        title=f"Results for: [italic]\"{query}\"[/italic]",
        border_style="blue",
        show_lines=True,
        expand=True,
    )
    t.add_column("#",        style="cyan",  width=4,   no_wrap=True)
    t.add_column("Dist",     style="white", width=7,   no_wrap=True)
    t.add_column("Source",   style="cyan",  width=22,  no_wrap=True)
    t.add_column("Chunk",    style="dim",   width=8,   no_wrap=True)
    t.add_column("Words",    style="dim",   width=6,   no_wrap=True)
    t.add_column("Snippet",  style="white")

    for i, (doc, meta, dist, id_) in enumerate(zip(docs, metadatas, distances, ids), 1):
        colour  = score_colour(dist)
        snippet = textwrap.shorten(doc.replace("\n", " "), width=SNIPPET_LEN, placeholder="…")
        t.add_row(
            str(i),
            f"[{colour}]{dist:.4f}[/{colour}]",
            meta.get("source", "?"),
            str(meta.get("chunk_index", "?")),
            str(meta.get("word_count", "?")),
            snippet,
        )

    console.print(t)
    console.print()

    # ── Explain the distance metric ──────────────────────────────────────────
    if distances:
        best = distances[0]
        worst = distances[-1]
        label = (
            "[green]very good[/green]"   if best < 0.25
            else "[yellow]moderate[/yellow]" if best < 0.5
            else "[red]weak[/red]"
        )
        console.print(f"  Best match distance: [bold]{best:.4f}[/bold] ({label})  "
                      f"  Worst: {worst:.4f}")
        console.print(f"  [dim]cosine distance: 0 = identical direction, 1 = orthogonal, 2 = opposite[/dim]\n")


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def ask(
    query:      str         = typer.Argument(...,   help="Natural-language question"),
    source:     str | None  = typer.Option(None,    "--source",    "-s",
                                           help="Filter to a specific document (slug)"),
    min_words:  int | None  = typer.Option(None,    "--min-words", "-w",
                                           help="Only return chunks with ≥ N words"),
    top_k:      int         = typer.Option(5,       "--top-k",     "-k"),
    chunk_size: int         = typer.Option(512,     "--chunk-size"),
) -> None:
    """
    Semantic search with optional metadata filtering.

    Examples:
      python query.py ask "how do neural networks learn?"
      python query.py ask "attention mechanism" --source transformer_deep_learning_architecture
      python query.py ask "training data" --min-words 60 --top-k 8
    """
    where = build_where(source, min_words)

    console.print(Panel(
        f"[bold]Query:[/bold]  {query}\n"
        f"[dim]Source filter:  {source or 'none'}\n"
        f"Min-words filter: {min_words or 'none'}\n"
        f"Top-k:  {top_k}  |  Chunk size: {chunk_size}[/dim]",
        title="🔍 ChromaDB Query", border_style="blue",
    ))

    col     = get_collection(chunk_size)
    results = col.query(
        query_texts=[query],
        n_results=top_k,
        where=where,                             # None → no filter
        include=["documents", "metadatas", "distances"],
    )

    display_results(results, query)


@app.command()
def sources(chunk_size: int = typer.Option(512, "--chunk-size")) -> None:
    """List all source documents in the collection (useful for --source filter)."""
    col   = get_collection(chunk_size)
    count = col.count()
    peek  = col.get(limit=count, include=["metadatas"])

    seen: dict[str, dict] = {}
    for m in peek["metadatas"]:
        src = m.get("source", "?")
        if src not in seen:
            seen[src] = {"topic": m.get("topic", src), "chunks": 0, "date": m.get("fetch_date", "?")}
        seen[src]["chunks"] += 1

    t = Table(title=f"Documents in collection (chunk_size={chunk_size})",
              border_style="cyan", show_lines=False)
    t.add_column("Slug (use for --source)", style="cyan")
    t.add_column("Topic",                   style="white")
    t.add_column("Chunks",                  style="green", justify="right")
    t.add_column("Fetched",                 style="dim")

    for slug, info in sorted(seen.items()):
        t.add_row(slug, info["topic"], str(info["chunks"]), info["date"])

    console.print(t)
    console.print()


@app.command()
def peek(
    source:     str = typer.Argument(..., help="Document slug to inspect"),
    n:          int = typer.Option(5,   "--n",          help="Number of chunks to show"),
    chunk_size: int = typer.Option(512, "--chunk-size"),
) -> None:
    """Show the first N chunks of a document (sanity-check ingestion quality)."""
    col  = get_collection(chunk_size)
    data = col.get(
        where={"source": {"$eq": source}},
        limit=n,
        include=["documents", "metadatas"],
    )

    docs  = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not docs:
        console.print(f"[red]No chunks found for source='{source}'[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]First {len(docs)} chunks of '{source}'[/bold]\n")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        console.print(Panel(
            doc[:500] + ("…" if len(doc) > 500 else ""),
            title=f"Chunk {meta.get('chunk_index', i)}  |  {meta.get('word_count', '?')} words",
            border_style="dim",
        ))
    console.print()


if __name__ == "__main__":
    app()
