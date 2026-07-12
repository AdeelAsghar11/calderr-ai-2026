"""
compare.py — Compare all retrieval approaches  (Lab 3.3 · Thursday)

Runs the same query through four approaches and shows results side by side:
  1. Naive       — top-k semantic search only (what Wednesday built)
  2. Hybrid      — BM25 + semantic with RRF fusion
  3. Rerank      — hybrid retrieval + cross-encoder re-ranking
  4. Multiquery  — query variations + hybrid + rerank

Also measures retrieval accuracy using keyword matching (same method as chunk_eval.py).
This gives you a quantitative comparison to present at the Friday standup.

Commands
--------
  python compare.py run "what is backpropagation?"
  python compare.py run "BLEU score formula" --keywords "bleu,bilingual,precision,ngram"
  python compare.py benchmark    # run 10 preset queries, print full accuracy table
"""

from __future__ import annotations

import os
import textwrap
import time

import chromadb
import typer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hybrid_retriever import HybridRetriever, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from reranker import CrossEncoderReranker
from multiquery import MultiQueryPipeline

load_dotenv(find_dotenv())

app     = typer.Typer(help="Retrieval comparison — Lab 3.3", add_completion=False)
console = Console()

# ── 10 benchmark queries with answer keywords ─────────────────────────────────
BENCHMARK = [
    {"query": "what is backpropagation?",
     "keywords": ["gradient", "backpropagation", "weights", "error"]},
    {"query": "how does the attention mechanism work?",
     "keywords": ["attention", "query", "key", "value"]},
    {"query": "what is reinforcement learning from human feedback?",
     "keywords": ["human feedback", "reward", "alignment", "rlhf"]},
    {"query": "how does BM25 score documents?",
     "keywords": ["bm25", "term frequency", "idf", "keyword"]},
    {"query": "what is a convolutional neural network?",
     "keywords": ["convolutional", "convolution", "image", "filter"]},
    {"query": "how does retrieval augmented generation work?",
     "keywords": ["retrieval", "generation", "context", "document"]},
    {"query": "what is transfer learning?",
     "keywords": ["transfer", "pretrain", "fine-tun", "knowledge"]},
    {"query": "what is the vanishing gradient problem?",
     "keywords": ["vanishing", "gradient", "deep", "layer"]},
    {"query": "what is approximate nearest neighbour search?",
     "keywords": ["approximate", "nearest", "neighbour", "ann"]},
    {"query": "what is object detection in computer vision?",
     "keywords": ["object", "detection", "bounding", "locali"]},
]


# ── Naive retriever (semantic only, from our existing ChromaDB) ───────────────
def naive_retrieve(query: str, top_k: int = 5) -> list[Document]:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    col    = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    results = col.query(query_texts=[query], n_results=top_k,
                        include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# ── Accuracy check ────────────────────────────────────────────────────────────
def check_hit(docs: list[Document], keywords: list[str]) -> tuple[bool, str]:
    """Return (hit, matched_keyword)."""
    combined = " ".join(d.page_content for d in docs).lower()
    for kw in keywords:
        if kw.lower() in combined:
            return True, kw
    return False, ""


# ── Mini result table for one approach ───────────────────────────────────────
def mini_table(
    docs:   list[Document],
    title:  str,
    border: str,
    ms:     float,
    hit:    bool,
) -> Table:
    status = "[green]HIT ✓[/green]" if hit else "[red]MISS ✗[/red]"
    t = Table(
        title=f"{title}\n[dim]{ms:.0f}ms  {status}[/dim]",
        border_style=border, show_lines=True, expand=True,
    )
    t.add_column("#",       style="cyan", width=4)
    t.add_column("Source",  style="cyan", width=18, no_wrap=True)
    t.add_column("Snippet", style="white")

    for i, doc in enumerate(docs, 1):
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 120, placeholder="…")
        t.add_row(str(i), doc.metadata.get("source", "?"), snippet)

    return t


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def run(
    query:    str = typer.Argument(..., help="Query to test"),
    keywords: str = typer.Option("",  "--keywords", "-k",
                                  help="Comma-separated answer keywords for accuracy check"),
    top_k:    int = typer.Option(5,   "--top-k"),
) -> None:
    """
    Run one query through all 4 approaches and compare results side by side.
    """
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    console.print()
    console.print(Panel(
        f"[bold]{query}[/bold]\n"
        + (f"[dim]Keywords: {', '.join(kw_list)}[/dim]" if kw_list else "[dim]No keywords provided[/dim]"),
        title="🔬 Retrieval Comparison", border_style="blue",
    ))

    # ── 1. Naive ──────────────────────────────────────────────────────────────
    t0          = time.perf_counter()
    naive_docs  = naive_retrieve(query, top_k)
    t_naive     = (time.perf_counter() - t0) * 1000
    naive_hit, _ = check_hit(naive_docs, kw_list) if kw_list else (None, "")

    # ── 2. Hybrid ─────────────────────────────────────────────────────────────
    console.print("[dim]Loading hybrid retriever...[/dim]")
    retriever   = HybridRetriever()
    t0          = time.perf_counter()
    hybrid_docs = retriever.retrieve(query, top_k)
    t_hybrid    = (time.perf_counter() - t0) * 1000
    hybrid_hit, _ = check_hit(hybrid_docs, kw_list) if kw_list else (None, "")

    # ── 3. Hybrid + Rerank ────────────────────────────────────────────────────
    console.print("[dim]Loading cross-encoder...[/dim]")
    reranker     = CrossEncoderReranker()
    candidates   = retriever.retrieve(query, top_k=top_k * 3)
    t0           = time.perf_counter()
    ranked       = reranker.rerank(query, candidates, top_k=top_k)
    t_rerank     = (time.perf_counter() - t0) * 1000
    rerank_docs  = [doc for _, doc in ranked]
    rerank_hit, _ = check_hit(rerank_docs, kw_list) if kw_list else (None, "")

    # ── 4. Multi-query ────────────────────────────────────────────────────────
    console.print("[dim]Running multi-query (calls Groq for variations)...[/dim]")
    pipeline    = MultiQueryPipeline(n_variations=3, initial_k=8, final_k=top_k)
    t0          = time.perf_counter()
    mq_docs, _ = pipeline.retrieve(query)
    t_mq        = (time.perf_counter() - t0) * 1000
    mq_hit, _   = check_hit(mq_docs, kw_list) if kw_list else (None, "")

    # ── Display ───────────────────────────────────────────────────────────────
    console.print()
    console.print(Columns([
        mini_table(naive_docs,  "1. Naive (semantic)",     "dim",     t_naive,  naive_hit),
        mini_table(hybrid_docs, "2. Hybrid (BM25+vec)",    "yellow",  t_hybrid, hybrid_hit),
    ], equal=True, expand=True))

    console.print()
    console.print(Columns([
        mini_table(rerank_docs, "3. Hybrid + Re-rank",     "blue",    t_rerank, rerank_hit),
        mini_table(mq_docs,     "4. Multi-query + Rerank", "green",   t_mq,     mq_hit),
    ], equal=True, expand=True))

    # ── Summary ───────────────────────────────────────────────────────────────
    if kw_list:
        rows = [
            ("Naive",            t_naive,           naive_hit),
            ("Hybrid",           t_hybrid,          hybrid_hit),
            ("Hybrid + Rerank",  t_hybrid + t_rerank, rerank_hit),
            ("Multi-query",      t_mq,              mq_hit),
        ]
        console.print()
        st = Table(title="Summary", border_style="cyan", show_lines=False)
        st.add_column("Approach", style="white")
        st.add_column("Latency",  style="dim",   justify="right")
        st.add_column("Hit",      style="white", justify="center")

        for name, ms, hit in rows:
            hit_str = "[green]✓ HIT[/green]" if hit else "[red]✗ MISS[/red]"
            st.add_row(name, f"{ms:.0f}ms", hit_str)

        console.print(st)
    console.print()


@app.command()
def benchmark(top_k: int = typer.Option(5, "--top-k")) -> None:
    """
    Run all 10 benchmark queries through all 4 approaches.
    Prints a full accuracy comparison table.
    Requires GROQ_API_KEY for multi-query.
    """
    console.print()
    console.print(Panel(
        f"[bold]Benchmark: 10 queries × 4 approaches[/bold]\n"
        f"[dim]top-k={top_k}[/dim]",
        border_style="blue",
    ))

    retriever = HybridRetriever()
    reranker  = CrossEncoderReranker()
    pipeline  = MultiQueryPipeline(n_variations=3, initial_k=8, final_k=top_k)

    scores = {
        "Naive":   [],
        "Hybrid":  [],
        "Rerank":  [],
        "MQ+RR":   [],
    }

    for item in BENCHMARK:
        q  = item["query"]
        kw = item["keywords"]
        console.print(f"  [dim]Testing: {q[:55]}...[/dim]")

        naive_docs = naive_retrieve(q, top_k)
        hybrid_docs = retriever.retrieve(q, top_k)

        candidates  = retriever.retrieve(q, top_k * 3)
        rerank_docs = [doc for _, doc in reranker.rerank(q, candidates, top_k)]

        mq_docs, _ = pipeline.retrieve(q)

        scores["Naive"].append(check_hit(naive_docs,  kw)[0])
        scores["Hybrid"].append(check_hit(hybrid_docs, kw)[0])
        scores["Rerank"].append(check_hit(rerank_docs, kw)[0])
        scores["MQ+RR"].append(check_hit(mq_docs,     kw)[0])

    # Results table
    console.print()
    t = Table(title="Benchmark Results", border_style="cyan", show_lines=True)
    t.add_column("Query",   style="white",  min_width=40)
    t.add_column("Naive",   style="white",  justify="center", width=8)
    t.add_column("Hybrid",  style="white",  justify="center", width=8)
    t.add_column("Rerank",  style="white",  justify="center", width=8)
    t.add_column("MQ+RR",   style="white",  justify="center", width=8)

    for i, item in enumerate(BENCHMARK):
        row = [item["query"][:55] + ("…" if len(item["query"]) > 55 else "")]
        for approach in ["Naive", "Hybrid", "Rerank", "MQ+RR"]:
            hit = scores[approach][i]
            row.append("[green]✓[/green]" if hit else "[red]✗[/red]")
        t.add_row(*row)

    t.add_section()
    acc_row = ["[bold]Accuracy[/bold]"]
    for approach in ["Naive", "Hybrid", "Rerank", "MQ+RR"]:
        hits = sum(scores[approach])
        pct  = hits / len(BENCHMARK) * 100
        col  = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
        acc_row.append(f"[{col}][bold]{pct:.0f}%[/bold][/{col}]\n[dim]{hits}/{len(BENCHMARK)}[/dim]")
    t.add_row(*acc_row)

    console.print(t)
    console.print()


if __name__ == "__main__":
    app()
