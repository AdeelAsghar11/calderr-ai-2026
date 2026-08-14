"""
ingest.py — Step 2: Ingest documents into ChromaDB

Pipeline:
  Load .txt files  →  Split into chunks  →  Embed  →  Store in ChromaDB

ChromaDB handles the embedding automatically via its EmbeddingFunction interface.
Each chunk is stored with metadata: source, chunk_index, word_count, char_count,
fetch_date, topic, chunk_size_config.

One collection is created per chunk size so you can compare 256 / 512 / 1024
without overwriting each other.

Commands
--------
  python ingest.py run                        # ingest all docs, chunk_size=512
  python ingest.py run --chunk-size 256       # experiment with smaller chunks
  python ingest.py run --chunk-size 1024      # larger chunks
  python ingest.py run --reset                # wipe collection and re-ingest
  python ingest.py stats                      # show collection stats
  python ingest.py list-collections           # show all stored collections
"""

from __future__ import annotations

import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
# pyrefly: ignore [missing-import]
from rich.table import Table

app     = typer.Typer(help="ChromaDB ingestion pipeline — Lab 3.2", add_completion=False)
console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
DOCS_DIR        = Path("./docs")
CHROMA_PATH     = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # same model as Lab 3.1
COLLECTION_BASE = "wiki_docs"


# ── ChromaDB helpers ──────────────────────────────────────────────────────────
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def collection_name(chunk_size: int) -> str:
    return f"{COLLECTION_BASE}_chunk{chunk_size}"


def get_or_create(client: chromadb.PersistentClient, chunk_size: int) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection with an embedded SentenceTransformer.

    Passing an EmbeddingFunction to ChromaDB means we call collection.add(documents=[...])
    and ChromaDB computes the vectors itself — we never touch numpy here.

    hnsw:space=cosine  → uses cosine distance for similarity (same as Lab 3.1)
    """
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=collection_name(chunk_size),
        embedding_function=ef,
        metadata={"hnsw:space": "cosine", "chunk_size": chunk_size},
    )


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def run(
    docs_dir:     str  = typer.Option("./docs",  "--docs",       help="Folder with .txt files"),
    chunk_size:   int  = typer.Option(512,        "--chunk-size", help="Characters per chunk (256/512/1024)"),
    chunk_overlap:int  = typer.Option(50,         "--overlap",    help="Character overlap between chunks"),
    reset:        bool = typer.Option(False,      "--reset",      help="Wipe collection before ingesting"),
) -> None:
    """
    Load .txt files, split into chunks, embed, and store in ChromaDB.

    Chunk size controls the granularity vs context trade-off:
      256  — small, precise chunks; retrieval is targeted but may lack context
      512  — balanced default; works well for most RAG scenarios
      1024 — large chunks; more context per result but retrieval is noisier
    """
    docs_path = Path(docs_dir)
    txt_files = sorted(docs_path.glob("*.txt"))

    if not txt_files:
        console.print(f"[red]No .txt files in {docs_dir}. Run fetch_docs.py first.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Ingestion pipeline[/bold]")
    console.print(f"  Docs:       {len(txt_files)} files in {docs_dir}")
    console.print(f"  Chunk size: [yellow]{chunk_size}[/yellow] chars  (overlap: {chunk_overlap})")
    console.print(f"  Embedding:  {EMBEDDING_MODEL}")
    console.print(f"  DB path:    {CHROMA_PATH}")
    console.print()

    # ── Setup ──────────────────────────────────────────────────────────────────
    client     = get_client()
    cname      = collection_name(chunk_size)

    if reset:
        try:
            client.delete_collection(cname)
            console.print(f"[yellow]  Deleted collection: {cname}[/yellow]")
        except Exception:
            pass

    collection = get_or_create(client, chunk_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on paragraph → sentence → word → char (in order)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # ── Ingest loop ─────────────────────────────────────────────────────────────
    total_chunks = 0
    doc_summary  = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting", total=len(txt_files))

        for txt_path in txt_files:
            progress.update(task, description=f"[cyan]{txt_path.stem[:30]}[/cyan]")

            # Load text
            text = txt_path.read_text(encoding="utf-8")

            # Load metadata sidecar
            meta_path = txt_path.with_suffix(".json")
            file_meta = {}
            if meta_path.exists():
                file_meta = json.loads(meta_path.read_text())

            # Split into chunks
            chunks = splitter.split_text(text)

            # Build ChromaDB batch
            ids        = []
            documents  = []
            metadatas  = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{txt_path.stem}_c{i:04d}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    # ── Provenance ───────────────────────────────────────────────
                    "source":            txt_path.stem,
                    "topic":             file_meta.get("title", txt_path.stem),
                    "url":               file_meta.get("url", ""),
                    "fetch_date":        file_meta.get("fetch_date", "unknown")[:10],
                    # ── Chunk position ───────────────────────────────────────────
                    "chunk_index":       i,
                    "total_chunks":      len(chunks),
                    # ── Size stats (useful for filtering) ────────────────────────
                    "word_count":        len(chunk.split()),
                    "char_count":        len(chunk),
                    "chunk_size_config": chunk_size,
                })

            # upsert = add or overwrite if id already exists
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total_chunks += len(chunks)
            doc_summary.append((txt_path.stem, len(chunks), len(text.split())))
            progress.advance(task)

    # ── Summary ───────────────────────────────────────────────────────────────
    t = Table(title=f"Ingestion complete · collection: {cname}", border_style="green", show_lines=False)
    t.add_column("Document",       style="cyan")
    t.add_column("Doc words",      style="dim",   justify="right")
    t.add_column("Chunks created", style="green", justify="right")
    t.add_column("Avg chunk words",style="dim",   justify="right")

    for stem, n_chunks, n_words in doc_summary:
        avg = n_words // n_chunks if n_chunks else 0
        t.add_row(stem, f"{n_words:,}", str(n_chunks), str(avg))

    t.add_section()
    t.add_row("[bold]TOTAL[/bold]", "", f"[bold]{total_chunks}[/bold]", "")
    console.print(t)
    console.print(f"\n[green]✓ {total_chunks} chunks stored in ChromaDB[/green]\n")


@app.command()
def stats(
    chunk_size: int = typer.Option(512, "--chunk-size"),
) -> None:
    """Show statistics for a stored collection."""
    client  = get_client()
    cname   = collection_name(chunk_size)

    try:
        col = client.get_collection(cname)
    except Exception:
        console.print(f"[red]Collection '{cname}' not found. Run: python ingest.py run[/red]")
        raise typer.Exit(1)

    count = col.count()

    # Peek at metadata to show source distribution
    peek = col.get(limit=count, include=["metadatas"])
    sources: dict[str, int] = {}
    total_words = 0

    for m in peek["metadatas"]:
        src = m.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        total_words += m.get("word_count", 0)

    t = Table(title=f"Collection: {cname}", border_style="cyan", show_lines=False)
    t.add_column("Source",  style="cyan")
    t.add_column("Chunks",  style="green", justify="right")
    t.add_column("% of total", style="dim", justify="right")

    for src, n in sorted(sources.items(), key=lambda x: -x[1]):
        pct = n / count * 100
        t.add_row(src, str(n), f"{pct:.1f}%")

    t.add_section()
    t.add_row("[bold]TOTAL[/bold]", f"[bold]{count}[/bold]", "100%")
    console.print(t)

    console.print(f"\n  Chunk size config: [yellow]{chunk_size}[/yellow] chars")
    console.print(f"  Total stored words: {total_words:,}")
    console.print(f"  Avg words/chunk: {total_words // count if count else 0}")
    console.print()


@app.command()
def list_collections() -> None:
    """Show all collections currently stored in ChromaDB."""
    client = get_client()
    cols   = client.list_collections()

    if not cols:
        console.print("[yellow]No collections found.[/yellow]")
        return

    t = Table(title="ChromaDB Collections", border_style="blue", show_lines=False)
    t.add_column("Name",      style="cyan")
    t.add_column("Count",     style="green", justify="right")
    t.add_column("Metadata",  style="dim")

    for col in cols:
        c    = client.get_collection(col.name)
        meta = str(col.metadata or {})
        t.add_row(col.name, str(c.count()), meta[:60])

    console.print(t)
    console.print()


if __name__ == "__main__":
    app()
