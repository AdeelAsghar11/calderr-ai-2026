"""
reranker.py — Cross-Encoder Re-ranking  (Lab 3.3 · Thursday)

Two-stage retrieval architecture:

  Stage 1 — Bi-encoder retrieval (fast, approximate)
    Query → embed → find top-20 candidates from vector store
    Time: ~10ms   Accuracy: good but not perfect

  Stage 2 — Cross-encoder re-ranking (slow, precise)
    For each of the 20 candidates: score relevance of (query, passage) together
    Sort by cross-encoder score. Return top-5.
    Time: ~200ms  Accuracy: significantly better

The cross-encoder reads the query AND passage simultaneously — it can do
cross-attention between them. The bi-encoder encodes them independently.
Cross-attention catches nuances the bi-encoder misses.

Model: BAAI/bge-reranker-base (~278MB)
  — trained specifically for passage re-ranking
  — outputs a single relevance float per (query, passage) pair
  — higher score = more relevant

Commands
--------
  python reranker.py rerank "what is backpropagation?"
  python reranker.py rerank "attention mechanism transformers" --initial-k 20 --final-k 5
  python reranker.py compare "BLEU score formula"
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import textwrap
import time

import typer
from langchain_core.documents import Document
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hybrid_retriever import HybridRetriever

app     = typer.Typer(help="Cross-encoder re-ranker — Lab 3.3", add_completion=False)
console = Console()

RERANKER_MODEL = "BAAI/bge-reranker-base"


# ── Core class ────────────────────────────────────────────────────────────────
class CrossEncoderReranker:
    """
    Wraps BAAI/bge-reranker-base for passage re-ranking.

    Usage:
        reranker = CrossEncoderReranker()
        ranked   = reranker.rerank(query, candidates, top_k=5)
        # ranked is [(score, document), ...] sorted descending
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        from sentence_transformers import CrossEncoder
        console.print(f"[dim]Loading cross-encoder: {model_name}...[/dim]")
        self._model      = CrossEncoder(model_name)
        self.model_name  = model_name

    def rerank(
        self,
        query:     str,
        documents: list[Document],
        top_k:     int = 5,
    ) -> list[tuple[float, Document]]:
        """
        Score each (query, document) pair and return top_k sorted by score.

        CrossEncoder.predict() runs a forward pass for EACH pair — this is
        why you apply it to a small candidate set (20–50), not the full corpus.

        Returns list of (relevance_score, document) tuples, highest first.
        """
        if not documents:
            return []

        pairs  = [(query, doc.page_content) for doc in documents]
        t0     = time.perf_counter()
        scores = self._model.predict(pairs)
        self._latency_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(
            zip(scores.tolist(), documents),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[:top_k]


# ── Display helpers ───────────────────────────────────────────────────────────
def rerank_table(
    ranked: list[tuple[float, Document]],
    title:  str,
    border: str = "blue",
) -> Table:
    t = Table(title=title, border_style=border, show_lines=True, expand=True)
    t.add_column("#",       style="cyan",  width=4)
    t.add_column("Score",   style="green", width=8,  no_wrap=True)
    t.add_column("Source",  style="cyan",  width=18, no_wrap=True)
    t.add_column("Snippet", style="white")

    for i, (score, doc) in enumerate(ranked, 1):
        colour  = "bright_green" if score > 3 else "green" if score > 0 else "yellow" if score > -3 else "red"
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 180, placeholder="…")
        t.add_row(
            str(i),
            f"[{colour}]{score:.3f}[/{colour}]",
            doc.metadata.get("source", "?"),
            snippet,
        )
    return t


def retrieval_table(
    docs:   list[Document],
    title:  str,
    border: str = "dim",
) -> Table:
    t = Table(title=title, border_style=border, show_lines=True, expand=True)
    t.add_column("#",       style="cyan",  width=4)
    t.add_column("Source",  style="cyan",  width=18, no_wrap=True)
    t.add_column("Snippet", style="white")

    for i, doc in enumerate(docs, 1):
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 200, placeholder="…")
        t.add_row(str(i), doc.metadata.get("source", "?"), snippet)
    return t


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def rerank(
    query:     str = typer.Argument(...),
    initial_k: int = typer.Option(20, "--initial-k", help="Candidates from first-stage retrieval"),
    final_k:   int = typer.Option(5,  "--final-k",   help="Final results after re-ranking"),
) -> None:
    """
    Two-stage retrieval: hybrid retrieval → cross-encoder re-ranking.

    Stage 1: retrieve initial_k candidates with hybrid search
    Stage 2: re-rank all candidates, keep final_k
    """
    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold] {query}\n"
        f"[dim]Stage 1: hybrid → top {initial_k}  |  Stage 2: rerank → top {final_k}[/dim]",
        title="🔀 Hybrid + Re-rank", border_style="blue",
    ))

    # Stage 1 — hybrid retrieval
    t0        = time.perf_counter()
    retriever = HybridRetriever()
    candidates = retriever.retrieve(query, top_k=initial_k)
    t_retrieve = (time.perf_counter() - t0) * 1000
    console.print(f"  [dim]Stage 1 complete: {len(candidates)} candidates in {t_retrieve:.0f}ms[/dim]")

    # Stage 2 — cross-encoder re-ranking
    t0       = time.perf_counter()
    reranker = CrossEncoderReranker()
    ranked   = reranker.rerank(query, candidates, top_k=final_k)
    t_rerank = (time.perf_counter() - t0) * 1000
    console.print(f"  [dim]Stage 2 complete: re-ranked {len(candidates)} docs in {t_rerank:.0f}ms[/dim]\n")

    console.print(rerank_table(
        ranked,
        title=f"Re-ranked top {final_k}  [{t_retrieve:.0f}ms retrieve + {t_rerank:.0f}ms rerank]",
    ))
    console.print()


@app.command()
def compare(
    query:     str = typer.Argument(...),
    initial_k: int = typer.Option(10, "--initial-k"),
    final_k:   int = typer.Option(5,  "--final-k"),
) -> None:
    """
    Side-by-side: hybrid top-k vs hybrid + re-ranked top-k.
    Shows concretely which documents moved up or down after re-ranking.
    """
    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold] {query}\n[dim]Comparing hybrid-only vs hybrid+rerank[/dim]",
        title="⚖  Before vs After Re-ranking", border_style="magenta",
    ))

    # Stage 1
    retriever  = HybridRetriever()
    candidates = retriever.retrieve(query, top_k=initial_k)

    # Stage 2
    reranker = CrossEncoderReranker()
    ranked   = reranker.rerank(query, candidates, top_k=final_k)

    # Build before table (top-k from hybrid, no rerank)
    before_items = [(0.0, doc) for doc in candidates[:final_k]]

    before_table = rerank_table(
        before_items,
        title=f"Hybrid only (top {final_k} of {initial_k})",
        border="yellow",
    )
    after_table = rerank_table(
        ranked,
        title=f"After re-ranking (top {final_k} of {initial_k})",
        border="green",
    )

    console.print(Columns([before_table, after_table], equal=True, expand=True))

    # Movement analysis
    before_sources = [doc.page_content[:80] for _, doc in before_items]
    after_sources  = [doc.page_content[:80] for _, doc in ranked]

    promoted  = [s for s in after_sources  if s not in before_sources]
    demoted   = [s for s in before_sources if s not in after_sources]

    console.print()
    if promoted:
        console.print(f"  [green]Promoted into top-{final_k}:[/green]  {len(promoted)} new result(s)")
    if demoted:
        console.print(f"  [yellow]Demoted out of top-{final_k}:[/yellow] {len(demoted)} result(s) replaced")
    if not promoted and not demoted:
        console.print(f"  [dim]Same results, different order[/dim]")
    console.print()


if __name__ == "__main__":
    app()
