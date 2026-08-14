"""
hybrid_retriever.py — BM25 + Semantic Hybrid Search  (Lab 3.3 · Thursday)

Two retrievers running in parallel over the same corpus:
  BM25Retriever   — keyword matching via TF-IDF-like scoring
  Chroma          — semantic similarity via cosine distance

Fusion via Reciprocal Rank Fusion (RRF) inside LangChain EnsembleRetriever.
A result near the top of BOTH lists gets the highest combined score.

Commands
--------
  python hybrid_retriever.py search "what is backpropagation?"
  python hybrid_retriever.py search "nn.MultiheadAttention masking" --top-k 8
  python hybrid_retriever.py search "BLEU score formula" --weights 0.3 0.7
  python hybrid_retriever.py info
"""

from __future__ import annotations

import hashlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import textwrap
import time
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# pyrefly: ignore [missing-import]
from langchain_classic.retrievers import EnsembleRetriever
# pyrefly: ignore [missing-import]
from langchain_community.retrievers import BM25Retriever
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table

app     = typer.Typer(help="Hybrid retriever — Lab 3.3", add_completion=False)
console = Console()

# ── Paths — points at lab-3-2 database ───────────────────────────────────────
CHROMA_PATH     = Path("../lab-3-2-rag-pipeline/chroma_db")
COLLECTION_NAME = "wiki_docs_chunk512"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Core class ────────────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Combines BM25 (keyword) and dense vector (semantic) retrieval.

    BM25 handles exact term matching — product names, error codes, identifiers,
    technical terms that embedding models compress into general concepts.

    Semantic search handles meaning — synonyms, paraphrases, conceptual queries
    where the exact words in the document differ from the query.

    EnsembleRetriever fuses both ranked lists with Reciprocal Rank Fusion (RRF):
      score = Σ  1 / (k + rank_in_list_i)    (k=60, the standard constant)

    A document appearing near the top of BOTH lists scores highest.
    A document appearing only in one list still gets a reasonable score.
    """

    def __init__(
        self,
        chroma_path:      str  = str(CHROMA_PATH),
        collection_name:  str  = COLLECTION_NAME,
        embedding_model:  str  = EMBEDDING_MODEL,
        bm25_k:           int  = 10,
        vector_k:         int  = 10,
        bm25_weight:      float = 0.5,
        vector_weight:    float = 0.5,
    ) -> None:
        self.collection_name = collection_name
        self.bm25_k          = bm25_k
        self.vector_k        = vector_k

        console.print(f"[dim]Loading collection '{collection_name}' from {chroma_path}...[/dim]")

        # ── Load all documents from ChromaDB ──────────────────────────────────
        client = chromadb.PersistentClient(path=chroma_path)
        ef     = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        try:
            col = client.get_collection(name=collection_name, embedding_function=ef)
        except Exception:
            console.print(f"[red]Collection '{collection_name}' not found.[/red]")
            console.print(f"[yellow]Run: cd ../lab-3-2-rag-pipeline && python ingest.py run[/yellow]")
            raise typer.Exit(1)

        count = col.count()
        data  = col.get(limit=count, include=["documents", "metadatas"])
        texts     = data["documents"]
        metadatas = data["metadatas"]
        console.print(f"[dim]  Loaded {len(texts)} chunks for BM25 index[/dim]")

        # ── BM25 retriever (keyword) ──────────────────────────────────────────
        self._bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
        self._bm25.k = bm25_k

        # ── Vector retriever (semantic) ───────────────────────────────────────
        # LangChain's Chroma wrapper connects to the SAME persistent database
        # but provides the LangChain Retriever interface.
        hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        chroma_lc     = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=hf_embeddings,
        )
        self._vector = chroma_lc.as_retriever(search_kwargs={"k": vector_k})

        # ── Ensemble (RRF fusion) ─────────────────────────────────────────────
        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25, self._vector],
            weights=[bm25_weight, vector_weight],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Run hybrid retrieval and return top_k documents."""
        docs = self._ensemble.invoke(query)
        return docs[:top_k]

    def retrieve_with_sources(
        self, query: str, top_k: int = 5
    ) -> tuple[list[Document], list[Document], list[Document]]:
        """
        Return (hybrid_results, bm25_only, vector_only) for analysis.
        Useful for understanding what each retriever contributed.
        """
        hybrid = self.retrieve(query, top_k)

        self._bm25.k = top_k
        bm25_docs    = self._bm25.invoke(query)
        vector_docs  = self._vector.invoke(query)
        self._bm25.k = self._bm25.k  # restore

        def content_set(docs: list[Document]) -> set[str]:
            return {d.page_content[:100] for d in docs}

        hybrid_set = content_set(hybrid)
        bm25_set   = content_set(bm25_docs)
        vector_set = content_set(vector_docs)

        bm25_only   = [d for d in bm25_docs   if d.page_content[:100] not in hybrid_set | vector_set]
        vector_only = [d for d in vector_docs if d.page_content[:100] not in hybrid_set | bm25_set]

        return hybrid, bm25_only, vector_only


# ── Display ───────────────────────────────────────────────────────────────────
def doc_table(
    docs:   list[Document],
    title:  str,
    border: str = "blue",
) -> Table:
    t = Table(title=title, border_style=border, show_lines=True, expand=True)
    t.add_column("#",       style="cyan",  width=4)
    t.add_column("Source",  style="cyan",  width=22, no_wrap=True)
    t.add_column("Words",   style="dim",   width=6,  no_wrap=True)
    t.add_column("Snippet", style="white")

    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata or {}
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 200, placeholder="…")
        t.add_row(str(i), meta.get("source", "?"), str(meta.get("word_count", "?")), snippet)

    return t


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def search(
    query:         str   = typer.Argument(...),
    top_k:         int   = typer.Option(5,   "--top-k",      "-k"),
    bm25_weight:   float = typer.Option(0.5, "--bm25",       help="BM25 weight (0–1)"),
    vector_weight: float = typer.Option(0.5, "--vector",     help="Semantic weight (0–1)"),
    show_sources:  bool  = typer.Option(False, "--breakdown", help="Show BM25-only vs vector-only"),
) -> None:
    """Hybrid BM25 + semantic search."""
    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold] {query}\n"
        f"[dim]top-k={top_k}  BM25 weight={bm25_weight}  vector weight={vector_weight}[/dim]",
        title="⚡ Hybrid Search", border_style="blue",
    ))

    t0 = time.perf_counter()
    retriever = HybridRetriever(
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )

    if show_sources:
        hybrid, bm25_only, vector_only = retriever.retrieve_with_sources(query, top_k)
        # pyrefly: ignore [missing-import]
        from rich.columns import Columns
        tables = [
            doc_table(hybrid,      f"Hybrid top-{top_k} (RRF fusion)", "blue"),
            doc_table(bm25_only,   "BM25-only results",                 "yellow"),
            doc_table(vector_only, "Vector-only results",               "magenta"),
        ]
        console.print(Columns(tables[:2], equal=True, expand=True))
        console.print(tables[2])
    else:
        hybrid = retriever.retrieve(query, top_k)
        console.print(doc_table(hybrid, f"Hybrid results — {(time.perf_counter()-t0)*1000:.0f}ms"))

    console.print()


@app.command()
def info() -> None:
    """Show collection info."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col   = client.get_collection(COLLECTION_NAME)
        count = col.count()
        console.print(f"\n[green]Collection:[/green] {COLLECTION_NAME}  |  [green]{count}[/green] chunks\n")
    except Exception:
        console.print(f"[red]Collection not found at {CHROMA_PATH}[/red]")


if __name__ == "__main__":
    app()
