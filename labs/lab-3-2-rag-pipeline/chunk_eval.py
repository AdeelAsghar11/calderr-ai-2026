"""
chunk_eval.py — Retrieval accuracy evaluation  (Lab 3.2 · Wednesday)

Runs 20 Q&A pairs against three chunk sizes (256, 512, 1024) and three
top-k values (3, 5, 10). Measures retrieval accuracy — did the retrieved
chunks contain the keywords needed to answer the question?

This is a retrieval evaluation, not a generation evaluation. We are not
asking "did the LLM answer correctly?" We are asking "did the retriever
surface the right content?" Generation quality depends on retrieval quality,
so this is the right metric to optimise first.

The scoring is simple:
  HIT  — at least one retrieved chunk contains a keyword from the answer
  MISS — none of the retrieved chunks contain any answer keyword

Run:
  python chunk_eval.py run             # full evaluation
  python chunk_eval.py run --quick     # test one chunk size (512) only
  python chunk_eval.py question 3      # debug a single question
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chromadb
import typer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingest import CHROMA_PATH, EMBEDDING_MODEL, collection_name

app     = typer.Typer(help="RAG retrieval evaluation — Lab 3.2", add_completion=False)
console = Console()

# ── 20 Q&A pairs ─────────────────────────────────────────────────────────────
# Questions drawn from the 10 Wikipedia articles fetched by fetch_docs.py.
# answer_keywords: if ANY keyword appears in ANY retrieved chunk → HIT.
# Keywords are lowercase and matched case-insensitively.

QA_PAIRS = [
    # ── Artificial Intelligence (2) ──────────────────────────────────────────
    {
        "id": 1,
        "question": "What is artificial intelligence?",
        "answer_keywords": ["intelligence", "machines", "human", "reasoning"],
        "source_hint": "artificial_intelligence",
    },
    {
        "id": 2,
        "question": "What happened at the Dartmouth Conference?",
        "answer_keywords": ["dartmouth", "1956", "mccarthy", "artificial intelligence"],
        "source_hint": "artificial_intelligence",
    },

    # ── Machine Learning (2) ─────────────────────────────────────────────────
    {
        "id": 3,
        "question": "What is supervised learning?",
        "answer_keywords": ["supervised", "labeled", "training", "input"],
        "source_hint": "machine_learning",
    },
    {
        "id": 4,
        "question": "What is overfitting in machine learning?",
        "answer_keywords": ["overfitting", "generaliz", "training data", "test"],
        "source_hint": "machine_learning",
    },

    # ── Deep Learning (2) ────────────────────────────────────────────────────
    {
        "id": 5,
        "question": "What is backpropagation?",
        "answer_keywords": ["backpropagation", "gradient", "weights", "error"],
        "source_hint": "deep_learning",
    },
    {
        "id": 6,
        "question": "What is a convolutional neural network?",
        "answer_keywords": ["convolutional", "convolution", "image", "filters"],
        "source_hint": "deep_learning",
    },

    # ── NLP (2) ──────────────────────────────────────────────────────────────
    {
        "id": 7,
        "question": "What is tokenization in natural language processing?",
        "answer_keywords": ["token", "word", "subword", "text"],
        "source_hint": "natural_language_processing",
    },
    {
        "id": 8,
        "question": "What is sentiment analysis?",
        "answer_keywords": ["sentiment", "opinion", "positive", "negative"],
        "source_hint": "natural_language_processing",
    },

    # ── Large Language Models (2) ─────────────────────────────────────────────
    {
        "id": 9,
        "question": "What is a large language model?",
        "answer_keywords": ["language model", "parameters", "transformer", "text"],
        "source_hint": "large_language_model",
    },
    {
        "id": 10,
        "question": "What is reinforcement learning from human feedback?",
        "answer_keywords": ["human feedback", "reward", "alignment", "rlhf"],
        "source_hint": "large_language_model",
    },

    # ── Transformers (2) ─────────────────────────────────────────────────────
    {
        "id": 11,
        "question": "How does the attention mechanism work?",
        "answer_keywords": ["attention", "query", "key", "value"],
        "source_hint": "transformer_deep_learning_architecture",
    },
    {
        "id": 12,
        "question": "What is self-attention in transformer models?",
        "answer_keywords": ["self-attention", "self attention", "sequence", "position"],
        "source_hint": "transformer_deep_learning_architecture",
    },

    # ── Vector Database (2) ──────────────────────────────────────────────────
    {
        "id": 13,
        "question": "What is a vector database?",
        "answer_keywords": ["vector", "embedding", "similarity", "search"],
        "source_hint": "vector_database",
    },
    {
        "id": 14,
        "question": "What is approximate nearest neighbor search?",
        "answer_keywords": ["approximate", "nearest", "neighbor", "ann"],
        "source_hint": "vector_database",
    },

    # ── RAG (2) ──────────────────────────────────────────────────────────────
    {
        "id": 15,
        "question": "What is retrieval augmented generation?",
        "answer_keywords": ["retrieval", "generation", "knowledge", "document"],
        "source_hint": "retrieval_augmented_generation",
    },
    {
        "id": 16,
        "question": "What problem does RAG solve in language models?",
        "answer_keywords": ["hallucination", "knowledge", "factual", "grounding"],
        "source_hint": "retrieval_augmented_generation",
    },

    # ── Reinforcement Learning (2) ────────────────────────────────────────────
    {
        "id": 17,
        "question": "What is the reward function in reinforcement learning?",
        "answer_keywords": ["reward", "agent", "environment", "action"],
        "source_hint": "reinforcement_learning",
    },
    {
        "id": 18,
        "question": "What is the difference between model-based and model-free RL?",
        "answer_keywords": ["model-based", "model-free", "environment", "transition"],
        "source_hint": "reinforcement_learning",
    },

    # ── Computer Vision (2) ──────────────────────────────────────────────────
    {
        "id": 19,
        "question": "What is image segmentation in computer vision?",
        "answer_keywords": ["segmentation", "pixel", "region", "object"],
        "source_hint": "computer_vision",
    },
    {
        "id": 20,
        "question": "What is object detection?",
        "answer_keywords": ["object", "detection", "bounding", "classify"],
        "source_hint": "computer_vision",
    },
]


# ── Evaluation logic ──────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    question_id: int
    question:    str
    hit:         bool
    best_dist:   float
    matched_kw:  str = ""

@dataclass
class RunResult:
    chunk_size:  int
    top_k:       int
    results:     list[EvalResult] = field(default_factory=list)
    latency_ms:  float = 0.0

    @property
    def hits(self)    -> int:   return sum(r.hit for r in self.results)
    @property
    def accuracy(self) -> float: return self.hits / len(self.results) * 100 if self.results else 0


def get_collection(chunk_size: int) -> chromadb.Collection | None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    cname  = collection_name(chunk_size)
    try:
        return client.get_collection(name=cname, embedding_function=ef)
    except Exception:
        return None


def evaluate_one(
    col:     chromadb.Collection,
    qa:      dict,
    top_k:   int,
) -> EvalResult:
    """
    Query ChromaDB for a question and check if any retrieved chunk
    contains any of the answer keywords.
    """
    results = col.query(
        query_texts=[qa["question"]],
        n_results=top_k,
        include=["documents", "distances"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]
    best_dist = distances[0] if distances else 1.0

    # Check all retrieved chunks for any keyword
    combined = " ".join(docs).lower()
    matched  = ""
    hit      = False
    for kw in qa["answer_keywords"]:
        if kw.lower() in combined:
            hit     = True
            matched = kw
            break

    return EvalResult(
        question_id=qa["id"],
        question=qa["question"],
        hit=hit,
        best_dist=best_dist,
        matched_kw=matched,
    )


def run_evaluation(chunk_size: int, top_k: int) -> RunResult | None:
    col = get_collection(chunk_size)
    if col is None:
        return None

    run    = RunResult(chunk_size=chunk_size, top_k=top_k)
    t0     = time.perf_counter()

    for qa in QA_PAIRS:
        result = evaluate_one(col, qa, top_k)
        run.results.append(result)

    run.latency_ms = (time.perf_counter() - t0) * 1000
    return run


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def run(
    quick: bool = typer.Option(False, "--quick", help="Only test chunk_size=512"),
) -> None:
    """
    Run the full evaluation: 20 questions × 3 chunk sizes × 3 top-k values.
    Requires all three collections to exist (run ingest.py for each chunk size first).
    """
    chunk_sizes = [512] if quick else [256, 512, 1024]
    top_ks      = [5] if quick else [3, 5, 10]

    console.print()
    console.print(Panel(
        f"[bold]Retrieval Accuracy Evaluation[/bold]\n"
        f"Questions: {len(QA_PAIRS)}  |  Chunk sizes: {chunk_sizes}  |  k values: {top_ks}",
        border_style="blue",
    ))

    # ── Run all combinations ──────────────────────────────────────────────────
    all_runs: dict[tuple, RunResult] = {}

    for cs in chunk_sizes:
        for k in top_ks:
            console.print(f"  [dim]Evaluating chunk_size={cs}, k={k}...[/dim]", end="")
            result = run_evaluation(cs, k)
            if result is None:
                console.print(f" [red]collection not found (run ingest.py --chunk-size {cs})[/red]")
            else:
                all_runs[(cs, k)] = result
                console.print(f" [green]{result.hits}/{len(QA_PAIRS)} ({result.accuracy:.0f}%)[/green]  {result.latency_ms:.0f}ms")

    if not all_runs:
        console.print("[red]No collections found. Run ingest.py for each chunk size first.[/red]")
        raise typer.Exit(1)

    console.print()

    # ── Table 1: Per-question breakdown (fixed k=5) ───────────────────────────
    k_display = 5 if 5 in top_ks else top_ks[0]
    t = Table(
        title=f"Per-question results  (k={k_display})",
        border_style="cyan", show_lines=True, expand=True,
    )
    t.add_column("#",        style="dim",   width=4)
    t.add_column("Question", style="white", min_width=35)
    for cs in chunk_sizes:
        key = (cs, k_display)
        label = f"chunk{cs}" + (" [green]✓[/green]" if key in all_runs else " [red]✗[/red]")
        t.add_column(label, justify="center", width=12)

    for i, qa in enumerate(QA_PAIRS):
        row = [str(qa["id"]), qa["question"][:55] + ("…" if len(qa["question"]) > 55 else "")]
        for cs in chunk_sizes:
            key = (cs, k_display)
            if key not in all_runs:
                row.append("[dim]n/a[/dim]")
                continue
            r = all_runs[key].results[i]
            row.append("[green]HIT[/green]" if r.hit else "[red]MISS[/red]")
        t.add_row(*row)

    t.add_section()
    acc_row = ["", "[bold]Accuracy[/bold]"]
    for cs in chunk_sizes:
        key = (cs, k_display)
        if key in all_runs:
            r   = all_runs[key]
            col = "green" if r.accuracy >= 80 else "yellow" if r.accuracy >= 60 else "red"
            acc_row.append(f"[{col}][bold]{r.accuracy:.0f}%[/bold] ({r.hits}/{len(QA_PAIRS)})[/{col}]")
        else:
            acc_row.append("[dim]n/a[/dim]")
    t.add_row(*acc_row)
    console.print(t)
    console.print()

    # ── Table 2: k comparison (fixed chunk_size=512) ──────────────────────────
    if not quick:
        cs_display = 512 if 512 in chunk_sizes else chunk_sizes[0]
        t2 = Table(
            title=f"Effect of top-k  (chunk_size={cs_display})",
            border_style="magenta", show_lines=False,
        )
        t2.add_column("top-k",     style="cyan",  width=8)
        t2.add_column("Hits",      style="green", width=10)
        t2.add_column("Accuracy",  style="white", width=12)
        t2.add_column("Latency",   style="dim",   width=12)
        t2.add_column("Insight",   style="dim")

        for k in top_ks:
            key = (cs_display, k)
            if key not in all_runs:
                t2.add_row(str(k), "n/a", "n/a", "n/a", "")
                continue
            r      = all_runs[key]
            colour = "green" if r.accuracy >= 80 else "yellow"
            insight = (
                "Few chunks — fast but misses some answers"    if k == 3
                else "Balanced — good default"                  if k == 5
                else "More context — higher recall, more noise" if k == 10
                else ""
            )
            t2.add_row(
                str(k),
                str(r.hits),
                f"[{colour}]{r.accuracy:.0f}%[/{colour}]",
                f"{r.latency_ms:.0f}ms",
                insight,
            )
        console.print(t2)
        console.print()

    # ── Table 3: Chunk size comparison (fixed k=5) ────────────────────────────
    if not quick:
        t3 = Table(
            title=f"Effect of chunk size  (k={k_display})",
            border_style="yellow", show_lines=False,
        )
        t3.add_column("Chunk size",  style="cyan",  width=12)
        t3.add_column("Accuracy",    style="green", width=12)
        t3.add_column("Insight",     style="dim")

        insights = {
            256: "Small chunks — precise retrieval, may lack context",
            512: "Balanced — good default for most RAG scenarios",
            1024: "Large chunks — rich context but noisier embeddings",
        }
        for cs in chunk_sizes:
            key = (cs, k_display)
            if key not in all_runs:
                t3.add_row(str(cs), "n/a", "")
                continue
            r      = all_runs[key]
            colour = "green" if r.accuracy >= 80 else "yellow" if r.accuracy >= 60 else "red"
            t3.add_row(
                f"{cs} chars",
                f"[{colour}]{r.accuracy:.0f}% ({r.hits}/{len(QA_PAIRS)})[/{colour}]",
                insights.get(cs, ""),
            )
        console.print(t3)
        console.print()

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_runs:
        best  = max(all_runs.values(), key=lambda r: r.accuracy)
        worst = min(all_runs.values(), key=lambda r: r.accuracy)
        console.print(Panel(
            f"[bold]Best:[/bold]   chunk_size={best.chunk_size}, k={best.top_k}  → "
            f"[green]{best.accuracy:.0f}%[/green] ({best.hits}/{len(QA_PAIRS)})\n"
            f"[bold]Worst:[/bold]  chunk_size={worst.chunk_size}, k={worst.top_k}  → "
            f"[red]{worst.accuracy:.0f}%[/red] ({worst.hits}/{len(QA_PAIRS)})",
            title="📊 Summary", border_style="green",
        ))
    console.print()


@app.command()
def question(
    question_id: int = typer.Argument(..., help="Question ID (1–20) to debug"),
    chunk_size:  int = typer.Option(512, "--chunk-size"),
    top_k:       int = typer.Option(5,   "--top-k"),
) -> None:
    """
    Debug a single question: show the retrieved chunks and whether keywords matched.
    Useful for understanding why a particular question is a HIT or MISS.
    """
    qa_map = {qa["id"]: qa for qa in QA_PAIRS}
    if question_id not in qa_map:
        console.print(f"[red]Question ID must be 1–{len(QA_PAIRS)}[/red]")
        raise typer.Exit(1)

    qa  = qa_map[question_id]
    col = get_collection(chunk_size)
    if col is None:
        console.print(f"[red]Collection for chunk_size={chunk_size} not found.[/red]")
        raise typer.Exit(1)

    results = col.query(
        query_texts=[qa["question"]],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    console.print()
    console.print(Panel(
        f"[bold]Q{qa['id']}:[/bold] {qa['question']}\n"
        f"[dim]Keywords: {', '.join(qa['answer_keywords'])}[/dim]",
        border_style="blue",
    ))

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        # Highlight matching keywords in the chunk
        text   = doc
        colour = "green" if dist < 0.35 else "yellow" if dist < 0.55 else "red"
        found  = [kw for kw in qa["answer_keywords"] if kw.lower() in doc.lower()]

        console.print(Panel(
            text[:600] + ("…" if len(text) > 600 else ""),
            title=f"Chunk {i}  |  dist={dist:.4f}  |  source={meta.get('source', '?')}  "
                  f"|  {'[green]KEYWORD MATCH: ' + ', '.join(found) + '[/green]' if found else '[red]no match[/red]'}",
            border_style=colour,
        ))

    result = evaluate_one(col, qa, top_k)
    status = f"[green]HIT — matched '{result.matched_kw}'[/green]" if result.hit else "[red]MISS — no keywords found in any chunk[/red]"
    console.print(f"\nResult: {status}\n")


@app.command()
def list_questions() -> None:
    """Show all 20 Q&A pairs."""
    t = Table(title="20 Evaluation Questions", border_style="cyan", show_lines=False)
    t.add_column("ID",       style="cyan",  width=4)
    t.add_column("Question", style="white")
    t.add_column("Keywords", style="dim")

    for qa in QA_PAIRS:
        t.add_row(str(qa["id"]), qa["question"], ", ".join(qa["answer_keywords"][:3]))

    console.print(t)
    console.print()


if __name__ == "__main__":
    app()
