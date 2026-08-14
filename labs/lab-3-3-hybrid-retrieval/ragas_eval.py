"""
ragas_eval.py — RAGAS Evaluation  (Lab 3.3 · Friday)

Loads eval_dataset.json and computes four RAG quality metrics:

  faithfulness          — are all answer claims supported by retrieved context?
  answer_relevancy      — does the answer actually address the question?
  context_precision     — of retrieved chunks, how many were actually relevant?
  context_recall        — did retrieval find all information needed to answer?

Uses Groq (llama-3.3-70b-versatile) as the evaluator LLM.
Uses HuggingFace embeddings (all-MiniLM-L6-v2) for answer_relevancy.

RAGAS calls the evaluator LLM ~4-6 times per question per metric.
For 20 questions this is ~400+ LLM calls — takes 5-10 minutes on Groq free tier.
Use --subset 5 for a fast test.

Commands
--------
  python ragas_eval.py run                    # full evaluation
  python ragas_eval.py run --subset 5         # quick 5-question test
  python ragas_eval.py run --no-recall        # skip context_recall (no ground_truth needed)
  python ragas_eval.py report                 # print saved results
  python ragas_eval.py analyse                # show what's low and what to fix
"""

from __future__ import annotations

import json
import os
import sys
import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Mocking modules to prevent Ragas import errors on newer langchain versions
try:
    # pyrefly: ignore [missing-import]
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    mock_chat_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    mock_chat_vertex.ChatVertexAI = object
    sys.modules["langchain_community.chat_models.vertexai"] = mock_chat_vertex

try:
    # pyrefly: ignore [missing-import]
    import langchain_community.llms.vertexai
except ModuleNotFoundError:
    mock_llms_vertex = types.ModuleType("langchain_community.llms.vertexai")
    mock_llms_vertex.VertexAI = object
    sys.modules["langchain_community.llms.vertexai"] = mock_llms_vertex

from pathlib import Path

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table

load_dotenv(find_dotenv())

app     = typer.Typer(help="RAGAS evaluation — Lab 3.3", add_completion=False)
console = Console()

DATASET_FILE = Path("eval_dataset.json")
RESULTS_FILE = Path("ragas_results.json")


# ── RAGAS setup ───────────────────────────────────────────────────────────────
def get_evaluator_llm():
    """Wrap Groq as a RAGAS-compatible evaluator LLM."""
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq
    # pyrefly: ignore [missing-import]
    from ragas.llms import LangchainLLMWrapper

    key = os.getenv("GROQ_API_KEY")
    if not key:
        console.print("[red]GROQ_API_KEY not found in .env[/red]")
        raise typer.Exit(1)

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=key)
    return LangchainLLMWrapper(llm)


def get_evaluator_embeddings():
    """Wrap HuggingFace embeddings as RAGAS-compatible embeddings."""
    # pyrefly: ignore [missing-import]
    from langchain_huggingface import HuggingFaceEmbeddings
    # pyrefly: ignore [missing-import]
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf)


def build_ragas_dataset(data: list[dict], include_recall: bool = True):
    """
    Convert our JSON dataset into the format RAGAS expects.
    Handles both RAGAS 0.1.x (datasets.Dataset) and 0.2.x (EvaluationDataset).
    """
    try:
        # ── Try RAGAS 0.2.x API ───────────────────────────────────────────────
        # pyrefly: ignore [missing-import]
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        samples = []
        for row in data:
            samples.append(SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=row.get("ground_truth", "") if include_recall else None,
            ))
        console.print("[dim]Using RAGAS 0.2.x dataset format[/dim]")
        return EvaluationDataset(samples=samples), "v2"

    except ImportError:
        # ── Fall back to RAGAS 0.1.x API ─────────────────────────────────────
        # pyrefly: ignore [missing-import]
        from datasets import Dataset

        payload = {
            "question": [r["question"] for r in data],
            "answer":   [r["answer"]   for r in data],
            "contexts": [r["contexts"] for r in data],
        }
        if include_recall:
            payload["ground_truth"] = [r.get("ground_truth", "") for r in data]

        console.print("[dim]Using RAGAS 0.1.x dataset format[/dim]")
        return Dataset.from_dict(payload), "v1"


def get_metrics(include_recall: bool, api_version: str, evaluator_llm, evaluator_embeddings):
    """Return metric objects configured for the detected RAGAS version."""
    if api_version == "v2":
        # pyrefly: ignore [missing-import]
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
        )
        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        ]
        if include_recall:
            # pyrefly: ignore [missing-import]
            from ragas.metrics import LLMContextRecall
            metrics.append(LLMContextRecall(llm=evaluator_llm))
    else:
        # pyrefly: ignore [missing-import]
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        faithfulness.llm       = evaluator_llm
        answer_relevancy.llm   = evaluator_llm
        answer_relevancy.embeddings = evaluator_embeddings
        context_precision.llm  = evaluator_llm
        metrics = [faithfulness, answer_relevancy, context_precision]

        if include_recall:
            # pyrefly: ignore [missing-import]
            from ragas.metrics import context_recall
            context_recall.llm = evaluator_llm
            metrics.append(context_recall)

    return metrics


# ── Display helpers ───────────────────────────────────────────────────────────
METRIC_NAMES = {
    "faithfulness":                              "Faithfulness",
    "answer_relevancy":                          "Answer Relevancy",
    "llm_context_precision_without_reference":   "Context Precision",
    "context_precision":                         "Context Precision",
    "llm_context_recall":                        "Context Recall",
    "context_recall":                            "Context Recall",
}

def score_colour(score: float) -> str:
    if score >= 0.85: return "bright_green"
    if score >= 0.70: return "green"
    if score >= 0.55: return "yellow"
    return "red"


def print_results_table(results_df, questions: list[str]) -> None:
    """Print per-question scores as a rich table."""
    metric_cols = [c for c in results_df.columns if c in METRIC_NAMES]

    t = Table(title="RAGAS Scores — Per Question", border_style="cyan", show_lines=True)
    t.add_column("#",          style="dim",   width=4)
    t.add_column("Question",   style="white", min_width=35)
    for col in metric_cols:
        label = METRIC_NAMES.get(col, col)[:18]
        t.add_column(label, justify="center", width=14)

    agg: dict[str, list[float]] = {c: [] for c in metric_cols}

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        q   = questions[i - 1] if i <= len(questions) else "?"
        cells = [str(i), q[:50] + ("…" if len(q) > 50 else "")]
        for col in metric_cols:
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and val != val):  # NaN
                cells.append("[dim]n/a[/dim]")
            else:
                score = float(val)
                agg[col].append(score)
                colour = score_colour(score)
                cells.append(f"[{colour}]{score:.3f}[/{colour}]")
        t.add_row(*cells)

    # Aggregate row
    t.add_section()
    agg_row = ["", "[bold]AVERAGE[/bold]"]
    for col in metric_cols:
        if agg[col]:
            avg    = sum(agg[col]) / len(agg[col])
            colour = score_colour(avg)
            agg_row.append(f"[{colour}][bold]{avg:.3f}[/bold][/{colour}]")
        else:
            agg_row.append("[dim]n/a[/dim]")
    t.add_row(*agg_row)

    console.print(t)


def print_summary_panel(results_df) -> dict[str, float]:
    """Print aggregate scores and return them as a dict."""
    metric_cols = [c for c in results_df.columns if c in METRIC_NAMES]
    aggregates  = {}

    lines = []
    for col in metric_cols:
        vals  = [float(v) for v in results_df[col].dropna() if str(v) != "nan"]
        if not vals:
            continue
        avg   = sum(vals) / len(vals)
        label = METRIC_NAMES.get(col, col)
        colour = score_colour(avg)
        aggregates[col] = avg

        verdict = (
            "excellent"          if avg >= 0.85
            else "good"          if avg >= 0.70
            else "needs work"    if avg >= 0.55
            else "critical issue"
        )
        lines.append(f"  {label:<28} [{colour}]{avg:.3f}[/{colour}]  [dim]{verdict}[/dim]")

    console.print(Panel("\n".join(lines), title="📊 Aggregate Scores", border_style="green"))
    return aggregates


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def run(
    subset:     int  = typer.Option(0,     "--subset",     "-n",
                                    help="Only evaluate first N questions (0 = all)"),
    no_recall:  bool = typer.Option(False, "--no-recall",
                                    help="Skip context_recall (faster, no ground_truth needed)"),
) -> None:
    """
    Run RAGAS evaluation on the saved dataset.
    Make sure to run build_dataset.py first.
    """
    if not DATASET_FILE.exists():
        console.print("[red]eval_dataset.json not found. Run: python build_dataset.py build[/red]")
        raise typer.Exit(1)

    data = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    if subset > 0:
        data = data[:subset]

    include_recall = (not no_recall) and all(r.get("ground_truth") for r in data)
    if not no_recall and not include_recall:
        console.print("[yellow]Ground truths missing — context_recall will be skipped.[/yellow]")
        console.print("[yellow]Re-run build_dataset.py without --skip-gt to enable it.[/yellow]")

    console.print()
    console.print(Panel(
        f"[bold]RAGAS Evaluation[/bold]\n"
        f"Questions:      {len(data)}\n"
        f"Metrics:        faithfulness, answer_relevancy, context_precision"
        + (", context_recall" if include_recall else "") + "\n"
        f"Evaluator LLM:  llama-3.1-8b-instant (Groq)\n"
        f"Embeddings:     all-MiniLM-L6-v2 (local)\n\n"
        f"[dim]This will make ~{len(data) * (4 if include_recall else 3) * 5} LLM calls. "
        f"Estimated time: {len(data) * (4 if include_recall else 3) // 3}–{len(data) * (4 if include_recall else 3) // 2} minutes.[/dim]",
        border_style="blue",
    ))

    # Load evaluator components
    console.print("[dim]Loading evaluator LLM and embeddings...[/dim]")
    evaluator_llm        = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()

    # Build dataset
    ragas_dataset, api_version = build_ragas_dataset(data, include_recall)
    metrics = get_metrics(include_recall, api_version, evaluator_llm, evaluator_embeddings)

    console.print(f"[dim]Running evaluation with {len(metrics)} metrics...[/dim]")
    console.print("[dim](Progress shown in terminal — each dot is one metric call)[/dim]\n")

    # Run with limited concurrency to avoid Groq rate limits/timeouts
    # pyrefly: ignore [missing-import]
    from ragas import evaluate
    # pyrefly: ignore [missing-import]
    from ragas.run_config import RunConfig
    
    run_config = RunConfig(max_workers=2, timeout=180)
    result = evaluate(dataset=ragas_dataset, metrics=metrics, run_config=run_config)

    result_df = result.to_pandas()
    questions  = [r["question"] for r in data]

    console.print()
    print_results_table(result_df, questions)
    console.print()
    aggregates = print_summary_panel(result_df)
    console.print()

    # Save results
    save_data = {
        "aggregates":  aggregates,
        "per_question": result_df.to_dict(orient="records"),
        "questions":   questions,
        "n_questions": len(data),
        "metrics":     list(aggregates.keys()),
    }
    RESULTS_FILE.write_text(json.dumps(save_data, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]✓ Results saved → {RESULTS_FILE}[/green]\n")


@app.command()
def report() -> None:
    """Print the saved RAGAS results."""
    if not RESULTS_FILE.exists():
        console.print("[red]ragas_results.json not found. Run: python ragas_eval.py run[/red]")
        raise typer.Exit(1)

    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    agg  = data.get("aggregates", {})

    t = Table(title="RAGAS Results Summary", border_style="cyan", show_lines=False)
    t.add_column("Metric",   style="white")
    t.add_column("Score",    style="white", justify="center", width=10)
    t.add_column("Verdict",  style="dim")

    for col, score in agg.items():
        label   = METRIC_NAMES.get(col, col)
        colour  = score_colour(float(score))
        verdict = (
            "excellent"           if float(score) >= 0.85
            else "good"           if float(score) >= 0.70
            else "needs work"     if float(score) >= 0.55
            else "critical issue"
        )
        t.add_row(label, f"[{colour}]{float(score):.3f}[/{colour}]", verdict)

    console.print(t)
    console.print(f"\n[dim]Based on {data.get('n_questions', '?')} questions.[/dim]\n")


@app.command()
def analyse() -> None:
    """
    Read RAGAS results and print a diagnostic: what is low and how to fix it.
    Use this to prepare the Friday standup code review section.
    """
    if not RESULTS_FILE.exists():
        console.print("[red]ragas_results.json not found. Run ragas_eval.py run first.[/red]")
        raise typer.Exit(1)

    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    agg  = {k: float(v) for k, v in data.get("aggregates", {}).items()}

    lines = []

    # Faithfulness
    f_key = next((k for k in agg if "faithful" in k), None)
    if f_key:
        f = agg[f_key]
        if f < 0.70:
            lines.append(
                "[red]⚠  Faithfulness is low ({:.3f})[/red]\n"
                "   The LLM is introducing claims not in the retrieved context.\n"
                "   Fix: strengthen 'answer ONLY from context' in system prompt.\n"
                "        Lower temperature. Switch to a larger model.".format(f)
            )
        elif f < 0.85:
            lines.append(
                "[yellow]⚡ Faithfulness is moderate ({:.3f})[/yellow]\n"
                "   Occasional hallucination. Review questions where it fails.\n"
                "   Fix: add 'cite the source document for each claim' to prompt.".format(f)
            )
        else:
            lines.append(f"[green]✓  Faithfulness is strong ({f:.3f})[/green] — answers stay within context.")

    # Answer Relevancy
    ar_key = next((k for k in agg if "relevancy" in k or "relevance" in k), None)
    if ar_key:
        ar = agg[ar_key]
        if ar < 0.70:
            lines.append(
                "\n[red]⚠  Answer Relevancy is low ({:.3f})[/red]\n"
                "   Answers are not addressing the questions asked.\n"
                "   Fix: retrieval is surfacing off-topic chunks.\n"
                "        Improve metadata filtering or use source-specific filters.".format(ar)
            )
        elif ar < 0.85:
            lines.append(
                "\n[yellow]⚡ Answer Relevancy is moderate ({:.3f})[/yellow]\n"
                "   Some answers drift from the question.\n"
                "   Fix: add 'answer the specific question asked, not the general topic' to prompt.".format(ar)
            )
        else:
            lines.append(f"\n[green]✓  Answer Relevancy is strong ({ar:.3f})[/green] — answers stay on topic.")

    # Context Precision
    cp_key = next((k for k in agg if "precision" in k), None)
    if cp_key:
        cp = agg[cp_key]
        if cp < 0.65:
            lines.append(
                "\n[red]⚠  Context Precision is low ({:.3f})[/red]\n"
                "   More than 35%% of retrieved chunks are irrelevant noise.\n"
                "   Fix: reduce chunk size (try 256). Add cross-encoder re-ranking.\n"
                "        Use source filters to narrow search scope.".format(cp)
            )
        elif cp < 0.80:
            lines.append(
                "\n[yellow]⚡ Context Precision is moderate ({:.3f})[/yellow]\n"
                "   Some irrelevant chunks are reaching the LLM.\n"
                "   Fix: tune re-ranker initial_k and final_k. Try chunk_size=256.".format(cp)
            )
        else:
            lines.append(f"\n[green]✓  Context Precision is strong ({cp:.3f})[/green] — retrieved chunks are mostly relevant.")

    # Context Recall
    cr_key = next((k for k in agg if "recall" in k), None)
    if cr_key:
        cr = agg[cr_key]
        if cr < 0.70:
            lines.append(
                "\n[red]⚠  Context Recall is low ({:.3f})[/red]\n"
                "   Retrieval is missing information needed to answer questions.\n"
                "   Fix: increase k. Use multi-query to cast a wider net.\n"
                "        Check if relevant documents are in the corpus.".format(cr)
            )
        elif cr < 0.85:
            lines.append(
                "\n[yellow]⚡ Context Recall is moderate ({:.3f})[/yellow]\n"
                "   Missing some relevant information.\n"
                "   Fix: try multi-query retrieval. Increase final_k from 5 to 8.".format(cr)
            )
        else:
            lines.append(f"\n[green]✓  Context Recall is strong ({cr:.3f})[/green] — retrieval finds most relevant content.")

    console.print()
    console.print(Panel("\n".join(lines), title="🔍 Diagnostic Analysis", border_style="blue"))

    # Standup prep
    console.print(Panel(
        "[bold]Friday Standup — Code Review talking points:[/bold]\n\n"
        + "\n".join(
            f"  • {METRIC_NAMES.get(k, k)}: [bold]{v:.3f}[/bold]"
            for k, v in agg.items()
        )
        + "\n\n[dim]For each metric, be ready to explain:\n"
        "  1. What it measures\n"
        "  2. Why your score is what it is\n"
        "  3. What you would do to improve it[/dim]",
        border_style="yellow",
    ))
    console.print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        app(["run"])
    else:
        app()
