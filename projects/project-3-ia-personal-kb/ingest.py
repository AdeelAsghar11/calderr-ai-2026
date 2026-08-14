"""
ingest.py — Chunk, Embed, and Store Personal Documents

Pipeline: load_documents() [loader.py] → split into chunks → embed → ChromaDB

Uses its own ChromaDB collection ("personal_kb") and its own persistent
directory ("./kb_chroma_db"), completely separate from the CalderR lab
databases — this is your own knowledge base, not internship material.

Commands
--------
  python ingest.py run                # ingest everything in ./docs
  python ingest.py run --reset        # wipe and re-ingest from scratch
  python ingest.py stats              # show what's stored
"""

from __future__ import annotations

import chromadb
import typer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

# pyrefly: ignore [missing-import]
from loader import load_documents

app     = typer.Typer(help="Ingest personal documents into ChromaDB", add_completion=False)
console = Console()

CHROMA_PATH     = "./kb_chroma_db"
COLLECTION_NAME = "personal_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50


def get_collection(reset: bool = False) -> chromadb.Collection:
    """Get or create the personal_kb collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


@app.command()
def run(
    docs_dir: str  = typer.Option("./docs", "--docs", help="Folder containing your documents"),
    reset:    bool = typer.Option(False,    "--reset", help="Wipe the collection before ingesting"),
) -> None:
    """Load every document in docs_dir, chunk it, embed it, and store in ChromaDB."""
    console.print(f"\n[bold]Loading documents from {docs_dir}...[/bold]\n")
    documents = load_documents(docs_dir)

    if not documents:
        console.print("[red]No documents found. Check the --docs path.[/red]")
        raise typer.Exit(1)

    console.print(f"  Loaded {len(documents)} documents\n")

    collection = get_collection(reset=reset)
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0
    summary      = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting", total=len(documents))

        for doc in documents:
            progress.update(task, description=f"[cyan]{doc.metadata['filename'][:35]}[/cyan]")

            chunks = splitter.split_text(doc.page_content)
            ids, texts, metas = [], [], []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.metadata['source']}_c{i:04d}"
                ids.append(chunk_id)
                texts.append(chunk)
                metas.append({
                    "source":       doc.metadata["source"],
                    "filename":     doc.metadata["filename"],
                    "file_type":    doc.metadata["file_type"],
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "word_count":   len(chunk.split()),
                })

            collection.upsert(ids=ids, documents=texts, metadatas=metas)
            total_chunks += len(chunks)
            summary.append((doc.metadata["filename"], doc.metadata["file_type"], len(chunks), doc.metadata["word_count"]))
            progress.advance(task)

    # ── Summary table ─────────────────────────────────────────────────────────
    t = Table(title="Ingestion complete", border_style="green", show_lines=False)
    t.add_column("Document", style="cyan")
    t.add_column("Type",     style="dim",   width=6)
    t.add_column("Words",    style="dim",   justify="right")
    t.add_column("Chunks",   style="green", justify="right")

    for fname, ftype, n_chunks, n_words in summary:
        t.add_row(fname, ftype, f"{n_words:,}", str(n_chunks))

    t.add_section()
    t.add_row("[bold]TOTAL[/bold]", "", "", f"[bold]{total_chunks}[/bold]")
    console.print(t)
    console.print(f"\n[green]✓ {len(documents)} documents, {total_chunks} chunks stored in '{COLLECTION_NAME}'[/green]\n")


@app.command()
def stats() -> None:
    """Show what's currently stored in the knowledge base."""
    collection = get_collection()
    count      = collection.count()

    if count == 0:
        console.print("[yellow]Knowledge base is empty. Run: python ingest.py run[/yellow]")
        return

    data    = collection.get(limit=count, include=["metadatas"])
    sources: dict[str, int] = {}
    for m in data["metadatas"]:
        src = m.get("filename", "?")
        sources[src] = sources.get(src, 0) + 1

    t = Table(title=f"Personal Knowledge Base — {count} chunks total", border_style="cyan", show_lines=False)
    t.add_column("Document", style="cyan")
    t.add_column("Chunks",   style="green", justify="right")

    for src, n in sorted(sources.items(), key=lambda x: -x[1]):
        t.add_row(src, str(n))

    console.print(t)
    console.print(f"\n  {len(sources)} documents, {count} total chunks\n")


if __name__ == "__main__":
    app()
