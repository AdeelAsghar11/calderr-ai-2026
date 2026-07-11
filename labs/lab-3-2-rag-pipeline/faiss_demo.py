"""
faiss_demo.py — FAISS: IndexFlatL2 vs IndexIVFFlat

This script reuses the embeddings cached in Lab 3.1 (.cache/embeddings_minilm.npy)
to demonstrate the two most important FAISS index types:

  IndexFlatL2    — exact brute-force search over L2 (Euclidean) distance
  IndexIVFFlat   — approximate search using Inverted File Index clustering

Key lesson: FAISS is a LIBRARY, not a DATABASE.
  ✓ Extremely fast vector search
  ✓ Handles billions of vectors efficiently
  ✗ No metadata — you must maintain your own id → metadata mapping
  ✗ No persistence by default (you save/load the index manually)
  ✗ No filtering — you filter results after retrieval by metadata

ChromaDB wraps a FAISS-like index (HNSW) but adds metadata, persistence,
filtering, and a collection abstraction on top.

Run:
  python faiss_demo.py
  python faiss_demo.py --query "space exploration and rockets"
  python faiss_demo.py --compare  # run both indexes on same query, compare results
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app     = typer.Typer(help="FAISS index comparison — Lab 3.2", add_completion=False)
console = Console()

# Reuse Lab 3.1 assets
LAB31_DIR      = Path("../lab-3-1-semantic-search")
CACHE_FILE     = LAB31_DIR / ".cache" / "embeddings_minilm.npy"
SENTENCES_FILE = LAB31_DIR / "sentences.py"


# ── Load Lab 3.1 data ─────────────────────────────────────────────────────────
def load_lab31() -> tuple[np.ndarray, list[str]]:
    """Load cached embeddings and sentences from Lab 3.1."""
    if not CACHE_FILE.exists():
        console.print(f"[red]Cache not found: {CACHE_FILE}[/red]")
        console.print("[yellow]Run Lab 3.1 first: python main.py search 'test' to populate cache.[/yellow]")
        sys.exit(1)

    sys.path.insert(0, str(LAB31_DIR))
    from sentences import SENTENCES  # type: ignore

    embeddings = np.load(CACHE_FILE).astype("float32")  # FAISS requires float32
    console.print(f"[dim]Loaded {len(SENTENCES)} sentences, embeddings shape: {embeddings.shape}[/dim]")
    return embeddings, SENTENCES


# ── Embed a query ─────────────────────────────────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    """Embed a single query string using the same model as Lab 3.1."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec   = model.encode([query], normalize_embeddings=True).astype("float32")
    return vec  # shape: (1, 384)


# ── FAISS index builders ──────────────────────────────────────────────────────
def build_flat(embeddings: np.ndarray):
    """
    IndexFlatIP — exact inner product search on L2-normalised vectors.

    'Flat' means every vector is stored and compared directly.
    IP (inner product) == cosine similarity when vectors are normalised.

    Pros:  guaranteed exact results, simple, no training step
    Cons:  O(n·d) per query — slow for n > ~1M vectors
    """
    import faiss
    d     = embeddings.shape[1]          # 384
    index = faiss.IndexFlatIP(d)         # IP = inner product (cosine on normed vecs)
    index.add(embeddings)
    return index


def build_ivf(embeddings: np.ndarray, nlist: int = 10):
    """
    IndexIVFFlat — approximate search using Inverted File Index.

    Step 1 (train): k-means clusters the corpus into nlist Voronoi cells.
    Step 2 (add):   each vector is assigned to its nearest cluster centroid.
    Step 3 (query): only nprobe cells are searched instead of the full corpus.

    Pros:  much faster for large n (O(nprobe * cluster_size * d) not O(n*d))
    Cons:  approximate — may miss some true nearest neighbours
           requires training before adding vectors
           nlist should be sqrt(n) roughly; here with 100 vecs, 10 is fine
    """
    import faiss
    d          = embeddings.shape[1]
    quantizer  = faiss.IndexFlatIP(d)   # used to assign vectors to cells
    index      = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)              # k-means clustering
    index.add(embeddings)
    return index


# ── Display helpers ───────────────────────────────────────────────────────────
def results_table(
    scores: np.ndarray,
    indices: np.ndarray,
    sentences: list[str],
    title: str,
    border: str = "blue",
) -> Table:
    t = Table(title=title, border_style=border, show_lines=True)
    t.add_column("#",       style="cyan",  width=4)
    t.add_column("Score",   style="green", width=8)
    t.add_column("Idx",     style="dim",   width=5)
    t.add_column("Sentence",style="white")

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        colour = "bright_green" if score > 0.7 else "green" if score > 0.5 else "yellow"
        t.add_row(str(rank), f"[{colour}]{score:.4f}[/{colour}]", str(idx), sentences[idx])

    return t


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def search(
    query:   str = typer.Option("machine learning and neural networks", "--query", "-q"),
    top_k:   int = typer.Option(5, "--top-k", "-k"),
    nprobe:  int = typer.Option(3, "--nprobe", help="IVF cells to search (higher=more accurate)"),
    compare: bool = typer.Option(False, "--compare", help="Show both indexes side by side"),
) -> None:
    """
    Run FAISS search with IndexFlatIP and/or IndexIVFFlat.

    With only 100 sentences the difference in speed is negligible, but
    with 1M+ vectors IVF is 10–100x faster with ~95% recall.
    """
    embeddings, sentences = load_lab31()

    console.print(Panel(
        f"[bold]Query:[/bold] {query}\n[dim]top-k={top_k}  nprobe={nprobe}[/dim]",
        title="⚡ FAISS Search", border_style="blue",
    ))

    query_vec = embed_query(query)

    # ── IndexFlatIP ───────────────────────────────────────────────────────────
    console.print("\n[dim]Building IndexFlatIP...[/dim]")
    t0         = time.perf_counter()
    flat_index = build_flat(embeddings)
    t_build    = time.perf_counter() - t0

    t0            = time.perf_counter()
    flat_scores, flat_idx = flat_index.search(query_vec, top_k)
    t_flat_search = time.perf_counter() - t0

    if not compare:
        console.print(results_table(
            flat_scores, flat_idx, sentences,
            title=f"IndexFlatIP  [build {t_build*1000:.1f}ms | search {t_flat_search*1000:.2f}ms]",
        ))
    else:
        # ── IndexIVFFlat ──────────────────────────────────────────────────────
        console.print("[dim]Building IndexIVFFlat (trains k-means clusters)...[/dim]")
        t0        = time.perf_counter()
        ivf_index = build_ivf(embeddings, nlist=10)
        ivf_index.nprobe = nprobe
        t_train   = time.perf_counter() - t0

        t0           = time.perf_counter()
        ivf_scores, ivf_idx = ivf_index.search(query_vec, top_k)
        t_ivf_search = time.perf_counter() - t0

        # Side-by-side comparison
        from rich.columns import Columns
        flat_table = results_table(
            flat_scores, flat_idx, sentences,
            title=f"IndexFlatIP (exact)\n[dim]build {t_build*1000:.1f}ms | search {t_flat_search*1000:.3f}ms[/dim]",
            border="blue",
        )
        ivf_table = results_table(
            ivf_scores, ivf_idx, sentences,
            title=f"IndexIVFFlat (approx, nprobe={nprobe})\n[dim]build {t_train*1000:.1f}ms | search {t_ivf_search*1000:.3f}ms[/dim]",
            border="magenta",
        )
        console.print(Columns([flat_table, ivf_table], equal=True, expand=True))

        # Recall comparison
        flat_set = set(flat_idx[0].tolist())
        ivf_set  = set(ivf_idx[0].tolist())
        overlap  = flat_set & ivf_set
        recall   = len(overlap) / top_k * 100
        colour   = "green" if recall >= 80 else "yellow" if recall >= 60 else "red"

        console.print(Panel(
            f"[bold]Approximate Recall @ {top_k}:[/bold]  [{colour}]{recall:.0f}%[/{colour}]\n"
            f"Both agree on:  {len(overlap)}/{top_k} results\n\n"
            f"[dim]With 100 sentences this is almost always 100%. "
            f"With 1M+ vectors, IVF recall drops to ~95% but speed improves 50–100×.[/dim]",
            title="📊 Recall Analysis", border_style="yellow",
        ))


@app.command()
def explain() -> None:
    """Print a conceptual comparison of FAISS vs ChromaDB."""
    console.print(Panel(
        """[bold]FAISS vs ChromaDB — what each is for[/bold]

[cyan]FAISS (Facebook AI Similarity Search)[/cyan]
  A C++ library with Python bindings for pure vector search.
  You manage everything yourself:
    • Maintain a separate list[] to map index positions → text/metadata
    • Save the index to disk with faiss.write_index() (binary format)
    • No built-in filtering — filter AFTER retrieval in Python

  [green]Use FAISS when:[/green]
    • You need maximum speed at billion-scale
    • You're building a custom retrieval system
    • You control the full stack (no need for a DB abstraction)

  Index types:
    IndexFlatIP/L2   — exact, slow for large n, no training
    IndexIVFFlat     — approximate, fast, requires training (k-means)
    IndexHNSW        — graph-based, very fast, good recall (~95%), no training
    IndexPQ          — product quantisation, compresses vectors to save RAM

[cyan]ChromaDB[/cyan]
  A vector DATABASE built on top of an HNSW index.
  Adds a proper DB abstraction:
    • Metadata stored alongside vectors (source, date, page, etc.)
    • Filtering via where= clauses (applied before vector search)
    • Persistent storage to disk automatically
    • Collections (like tables) to organise different document sets
    • Python / HTTP / gRPC API

  [green]Use ChromaDB when:[/green]
    • You're building RAG pipelines
    • You need metadata filtering
    • You want persistence without manual index management
    • n < ~10M vectors (ChromaDB is not designed for web-scale)

  For web-scale production (100M+ vectors): use Qdrant, Pinecone, or Weaviate.
""",
        border_style="blue",
    ))


if __name__ == "__main__":
    app(["search", "--compare"])
