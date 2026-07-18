"""
kb.py — Personal Knowledge Base CLI

RAG over your own documents: resume, portfolio, GitHub project READMEs,
CalderR internship materials, hackathon writeups.

Commands
--------
  python kb.py ask "what was my BSL accuracy?"
  python kb.py ask "what did I build for HACKDATA?" --show-context
  python kb.py ask "what topics were in CalderR week 2?" --source calderr_week_2
  python kb.py chat
  python kb.py sources
"""

from __future__ import annotations

import os
import textwrap

import chromadb
import typer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv(find_dotenv())

app     = typer.Typer(help="Personal Knowledge Base — RAG CLI", add_completion=False)
console = Console()

# ── Prompt ────────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a personal knowledge assistant answering questions about the "
     "user's own documents: resume, portfolio, project READMEs, internship "
     "materials, and achievement writeups.\n"
     "Answer using ONLY the provided context. If the context doesn't contain "
     "the answer, say so honestly rather than guessing.\n"
     "Be direct and specific. Quote exact numbers, project names, and dates "
     "as they appear in the context."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        console.print("[red]Knowledge base not found. Run: python ingest.py run[/red]")
        raise typer.Exit(1)


def get_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        console.print("[red]GROQ_API_KEY not found in .env[/red]")
        raise typer.Exit(1)
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key)


def retrieve(
    collection: chromadb.Collection,
    question:   str,
    top_k:      int = 5,
    source:     str | None = None,
) -> tuple[list[str], list[dict], list[float]]:
    """Query ChromaDB, optionally filtered to one document source."""
    where = {"source": {"$eq": source}} if source else None
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def build_context(docs: list[str], metas: list[dict]) -> str:
    """Format retrieved chunks into a numbered, source-attributed context block."""
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        fname = meta.get("filename", meta.get("source", "unknown"))
        parts.append(f"[Source {i} — {fname}]\n{doc}")
    return "\n\n---\n\n".join(parts)


def show_sources_table(docs: list[str], metas: list[dict], distances: list[float]) -> None:
    """Print retrieved chunks before the answer — shows exactly what grounded it."""
    t = Table(title="Retrieved sources", border_style="dim", show_lines=True, expand=True)
    t.add_column("#",        style="cyan",  width=4)
    t.add_column("Dist",     style="green", width=7)
    t.add_column("Document", style="cyan",  width=32, no_wrap=True)
    t.add_column("Snippet",  style="white")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        colour  = "bright_green" if dist < 0.30 else "green" if dist < 0.50 else "yellow"
        snippet = textwrap.shorten(doc.replace("\n", " "), 150, placeholder="…")
        t.add_row(str(i), f"[{colour}]{dist:.3f}[/{colour}]", meta.get("filename", "?"), snippet)

    console.print(t)
    console.print()


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def ask(
    question:     str      = typer.Argument(..., help="Question about your own documents"),
    top_k:        int      = typer.Option(5,    "--top-k",      "-k"),
    source:       str|None = typer.Option(None, "--source",     "-s",
                                          help="Filter to one document (see: python kb.py sources)"),
    show_context: bool     = typer.Option(False,"--show-context", help="Show retrieved chunks + citations"),
) -> None:
    """Ask a question. Answer streams with source citations shown on request."""
    console.print()
    console.print(Panel(
        f"[bold]{question}[/bold]\n[dim]top_k={top_k}  source={source or 'all documents'}[/dim]",
        title="🔍 Personal Knowledge Base", border_style="blue",
    ))

    collection = get_collection()
    docs, metas, dists = retrieve(collection, question, top_k, source)

    if not docs:
        console.print("[yellow]No relevant chunks found for this query + filter.[/yellow]\n")
        return

    if show_context:
        show_sources_table(docs, metas, dists)

    context = build_context(docs, metas)
    llm     = get_llm()
    chain   = RAG_PROMPT | llm | StrOutputParser()

    console.print("[bold cyan]Answer:[/bold cyan]")
    for token in chain.stream({"context": context, "question": question}):
        console.print(token, end="", highlight=False)
    console.print("\n")

    # Always show which documents grounded the answer, even without --show-context
    cited = sorted({m.get("filename", "?") for m in metas})
    console.print(f"[dim]Sources: {', '.join(cited)}[/dim]\n")


@app.command()
def chat() -> None:
    """Interactive multi-turn chat over your personal knowledge base."""
    console.print(Panel(
        "[bold]Personal KB Chat[/bold]\n[dim]/context  toggle source display | /exit  quit[/dim]",
        border_style="magenta",
    ))

    collection = get_collection()
    llm        = get_llm()
    chain      = RAG_PROMPT | llm | StrOutputParser()
    show       = False

    while True:
        try:
            q = console.input("\n[bold magenta]You:[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q == "/exit":
            break
        if q == "/context":
            show = not show
            console.print(f"[dim]Context display: {'on' if show else 'off'}[/dim]")
            continue

        docs, metas, dists = retrieve(collection, q, top_k=5)
        if show:
            show_sources_table(docs, metas, dists)

        context = build_context(docs, metas)

        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        for token in chain.stream({"context": context, "question": q}):
            console.print(token, end="", highlight=False)
        console.print()

    console.print("\n[dim]Session ended.[/dim]")


@app.command()
def sources() -> None:
    """List all documents in the knowledge base — use the slug for --source filtering."""
    collection = get_collection()
    count      = collection.count()
    data       = collection.get(limit=count, include=["metadatas"])

    seen: dict[str, dict] = {}
    for m in data["metadatas"]:
        src = m.get("source", "?")
        if src not in seen:
            seen[src] = {"filename": m.get("filename", src), "chunks": 0}
        seen[src]["chunks"] += 1

    t = Table(title="Documents  (use the slug for --source)", border_style="cyan", show_lines=False)
    t.add_column("Slug",     style="cyan")
    t.add_column("Filename", style="white")
    t.add_column("Chunks",   style="green", justify="right")

    for slug, info in sorted(seen.items()):
        t.add_row(slug, info["filename"], str(info["chunks"]))

    console.print(t)
    console.print()


if __name__ == "__main__":
    app()
