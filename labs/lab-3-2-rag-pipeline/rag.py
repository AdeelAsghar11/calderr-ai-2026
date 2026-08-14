"""
rag.py — Naive RAG Pipeline  (Lab 3.2 · Wednesday)

Completes the pipeline started on Tuesday:
  ChromaDB retrieval  →  context formatting  →  Groq generation

The full flow for every query:
  1. Embed the user question (same model used during ingestion)
  2. Retrieve top-k most similar chunks from ChromaDB
  3. Format chunks into a context string
  4. Send context + question to Groq LLM via LangChain
  5. Stream the answer back

Commands
--------
  python rag.py ask "what is backpropagation?"
  python rag.py ask "how does attention work?" --show-context
  python rag.py ask "what is RLHF?" --chunk-size 256 --top-k 3
  python rag.py ask "explain vector databases" --source vector_database
  python rag.py chat           # interactive multi-turn mode
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from rich.columns import Columns
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table

# pyrefly: ignore [missing-import]
from ingest import CHROMA_PATH, EMBEDDING_MODEL, collection_name

# ── Load .env from anywhere up the directory tree ──────────────────────────────
load_dotenv(find_dotenv())

app     = typer.Typer(help="Naive RAG pipeline — Lab 3.2", add_completion=False)
console = Console()

# ── RAG prompt ────────────────────────────────────────────────────────────────
# This is the core of naive RAG. The LLM is instructed to answer ONLY from
# context — this grounds it and prevents hallucination.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant. Answer questions using ONLY the provided context.\n"
     "If the context does not contain enough information, say exactly: "
     "'The provided documents do not contain enough information to answer this.'\n"
     "Do not use any knowledge outside the context. Be concise."),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}\n\n"
     "Answer:"),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_collection(chunk_size: int) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    cname  = collection_name(chunk_size)
    try:
        return client.get_collection(name=cname, embedding_function=ef)
    except Exception:
        console.print(f"[red]Collection '{cname}' not found.[/red]")
        console.print(f"[yellow]Run first: python ingest.py run --chunk-size {chunk_size}[/yellow]")
        raise typer.Exit(1)


def get_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        console.print("[red]GROQ_API_KEY not found. Add it to your .env file.[/red]")
        raise typer.Exit(1)
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key)


def retrieve(
    collection: chromadb.Collection,
    question:   str,
    top_k:      int,
    source:     str | None = None,
    min_words:  int | None = None,
) -> tuple[list[str], list[dict], list[float]]:
    """Run a ChromaDB query and return (documents, metadatas, distances)."""
    conditions = []
    if source:
        conditions.append({"source": {"$eq": source}})
    if min_words:
        conditions.append({"word_count": {"$gte": min_words}})

    where = (
        None            if not conditions
        else conditions[0] if len(conditions) == 1
        else {"$and": conditions}
    )

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )


def build_context(docs: list[str], metas: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.
    Source attribution is included so the LLM can reference it.
    """
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        source = meta.get("topic", meta.get("source", "unknown"))
        parts.append(f"[Source {i} — {source}]\n{doc}")
    return "\n\n---\n\n".join(parts)


def show_retrieved(
    docs:      list[str],
    metas:     list[dict],
    distances: list[float],
) -> None:
    """Print a table of retrieved chunks before the answer."""
    t = Table(
        title="Retrieved chunks", border_style="dim",
        show_lines=True, expand=True,
    )
    t.add_column("#",        style="cyan",  width=4,  no_wrap=True)
    t.add_column("Dist",     style="green", width=7,  no_wrap=True)
    t.add_column("Source",   style="cyan",  width=22, no_wrap=True)
    t.add_column("Chunk",    style="dim",   width=6,  no_wrap=True)
    t.add_column("Snippet",  style="white")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        colour  = "bright_green" if dist < 0.25 else "green" if dist < 0.45 else "yellow"
        snippet = textwrap.shorten(doc.replace("\n", " "), 220, placeholder="…")
        t.add_row(
            str(i),
            f"[{colour}]{dist:.4f}[/{colour}]",
            meta.get("source", "?"),
            str(meta.get("chunk_index", "?")),
            snippet,
        )
    console.print(t)
    console.print()


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def ask(
    question:     str       = typer.Argument(..., help="Question to ask the RAG system"),
    chunk_size:   int       = typer.Option(512,   "--chunk-size", "-c"),
    top_k:        int       = typer.Option(5,     "--top-k",      "-k"),
    source:       str|None  = typer.Option(None,  "--source",     "-s",
                                           help="Limit retrieval to one document slug"),
    show_context: bool      = typer.Option(False, "--show-context",
                                           help="Print retrieved chunks before the answer"),
    no_stream:    bool      = typer.Option(False, "--no-stream",
                                           help="Print full answer at once instead of streaming"),
) -> None:
    """
    Ask a question. RAG retrieves relevant chunks then Groq generates an answer.

    Useful flags:
      --show-context    see exactly what context the LLM received
      --chunk-size 256  test a different chunk size collection
      --top-k 3         fewer chunks = less context but faster
      --source deep_learning   only retrieve from one document
    """
    console.print()
    console.print(Panel(
        f"[bold]{question}[/bold]\n"
        f"[dim]chunk_size={chunk_size}  top_k={top_k}  source={source or 'all'}[/dim]",
        title="🔍 RAG Query", border_style="blue",
    ))

    # ── Retrieve ───────────────────────────────────────────────────────────────
    col                  = get_collection(chunk_size)
    docs, metas, dists   = retrieve(col, question, top_k, source)

    if show_context:
        show_retrieved(docs, metas, dists)

    context = build_context(docs, metas)

    # ── Generate ───────────────────────────────────────────────────────────────
    llm   = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    console.print("[bold cyan]Answer:[/bold cyan]")

    if no_stream:
        answer = chain.invoke({"context": context, "question": question})
        console.print(answer)
    else:
        for token in chain.stream({"context": context, "question": question}):
            console.print(token, end="", highlight=False)
        console.print()

    console.print()


@app.command()
def chat(
    chunk_size: int = typer.Option(512, "--chunk-size", "-c"),
    top_k:      int = typer.Option(5,   "--top-k",     "-k"),
) -> None:
    """
    Interactive multi-turn chat. Type your questions one by one.
    Commands: /context  show retrieved chunks  |  /clear  reset  |  /exit  quit
    """
    console.print(Panel(
        f"[bold]RAG Chat[/bold] — chunk_size={chunk_size}  top_k={top_k}\n"
        "[dim]/context  show chunks | /exit  quit[/dim]",
        border_style="magenta",
    ))

    col   = get_collection(chunk_size)
    llm   = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    show  = False

    while True:
        try:
            question = console.input("\n[bold magenta]You:[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question == "/exit":
            break
        if question == "/context":
            show = not show
            console.print(f"[dim]Context display: {'on' if show else 'off'}[/dim]")
            continue
        if question == "/clear":
            console.clear()
            continue

        docs, metas, dists = retrieve(col, question, top_k)

        if show:
            show_retrieved(docs, metas, dists)

        context = build_context(docs, metas)

        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        for token in chain.stream({"context": context, "question": question}):
            console.print(token, end="", highlight=False)
        console.print()

    console.print("\n[dim]Session ended.[/dim]")


if __name__ == "__main__":
    app()
