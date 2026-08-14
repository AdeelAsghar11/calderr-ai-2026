"""
kb.py — Personal Knowledge Base CLI

RAG over your own documents: resume, portfolio, GitHub project READMEs,
CalderR internship materials, hackathon writeups.

Uses hybrid retrieval (BM25 + semantic, RRF fusion) instead of pure
semantic search. Pure semantic search missed exact-phrase content like
"RAGAS metrics: faithfulness, answer relevancy, context precision, context
recall" because a nearby chunk scored closer on embedding similarity than
the chunk actually containing the answer. BM25 catches this by direct
keyword match — the fusion keeps both strengths. top_k also bumped from
5 to 8 to give correct chunks more room to surface.

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

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table

# pyrefly: ignore [missing-import]
from hybrid import PersonalHybridRetriever
# pyrefly: ignore [missing-import]
from ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv(find_dotenv())

app     = typer.Typer(help="Personal Knowledge Base — RAG CLI", add_completion=False)
console = Console()

DEFAULT_TOP_K = 8   # bumped from 5 — gives correct chunks more room to surface

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
def get_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        console.print("[red]GROQ_API_KEY not found in .env[/red]")
        raise typer.Exit(1)
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key)


def build_context(docs: list[Document]) -> str:
    """Format retrieved chunks into a numbered, source-attributed context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        parts.append(f"[Source {i} — {fname}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def show_sources_table(docs: list[Document]) -> None:
    """Print retrieved chunks before the answer — shows exactly what grounded it."""
    t = Table(
        title="Retrieved sources (hybrid: BM25 + semantic)",
        border_style="dim", show_lines=True, expand=True,
    )
    t.add_column("#",        style="cyan",  width=4)
    t.add_column("Document", style="cyan",  width=32, no_wrap=True)
    t.add_column("Snippet",  style="white")

    for i, doc in enumerate(docs, 1):
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 150, placeholder="…")
        t.add_row(str(i), doc.metadata.get("filename", "?"), snippet)

    console.print(t)
    console.print()


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def ask(
    question:     str      = typer.Argument(..., help="Question about your own documents"),
    top_k:        int      = typer.Option(DEFAULT_TOP_K, "--top-k", "-k"),
    source:       str|None = typer.Option(None, "--source", "-s",
                                          help="Filter to one document (see: python kb.py sources)"),
    show_context: bool     = typer.Option(False, "--show-context", help="Show retrieved chunks + citations"),
) -> None:
    """Ask a question. Answer streams with source citations shown on request."""
    console.print()
    console.print(Panel(
        f"[bold]{question}[/bold]\n[dim]top_k={top_k}  source={source or 'all documents'}[/dim]",
        title="🔍 Personal Knowledge Base (hybrid)", border_style="blue",
    ))

    retriever = PersonalHybridRetriever()
    docs      = retriever.retrieve(question, top_k=top_k, source=source)

    if not docs:
        console.print("[yellow]No relevant chunks found for this query + filter.[/yellow]\n")
        return

    if show_context:
        show_sources_table(docs)

    context = build_context(docs)
    llm     = get_llm()
    chain   = RAG_PROMPT | llm | StrOutputParser()

    console.print("[bold cyan]Answer:[/bold cyan]")
    for token in chain.stream({"context": context, "question": question}):
        console.print(token, end="", highlight=False)
    console.print("\n")

    cited = sorted({d.metadata.get("filename", "?") for d in docs})
    console.print(f"[dim]Sources: {', '.join(cited)}[/dim]\n")


@app.command()
def chat() -> None:
    """Interactive multi-turn chat over your personal knowledge base."""
    console.print(Panel(
        "[bold]Personal KB Chat (hybrid retrieval)[/bold]\n[dim]/context  toggle source display | /exit  quit[/dim]",
        border_style="magenta",
    ))

    retriever = PersonalHybridRetriever()   # loaded once, reused across turns
    llm       = get_llm()
    chain     = RAG_PROMPT | llm | StrOutputParser()
    show      = False

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

        docs = retriever.retrieve(q, top_k=DEFAULT_TOP_K)
        if show:
            show_sources_table(docs)

        context = build_context(docs)

        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        for token in chain.stream({"context": context, "question": q}):
            console.print(token, end="", highlight=False)
        console.print()

    console.print("\n[dim]Session ended.[/dim]")


@app.command()
def sources() -> None:
    """List all documents in the knowledge base — use the slug for --source filtering."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    count = collection.count()
    data  = collection.get(limit=count, include=["metadatas"])

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