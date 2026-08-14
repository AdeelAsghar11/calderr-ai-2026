"""
main.py — Semantic Search CLI  (Lab 3.1)

Commands
--------
  search    <query>          — search using one model
  compare   <query>          — side-by-side model comparison
  visualize                  — PCA scatter plot saved to embeddings_pca.png
  info                       — corpus and cache status

Examples
--------
  python main.py search "space exploration and astronomy"
  python main.py search "machine learning" --model bge --top-k 8
  python main.py compare "history of ancient civilisations"
  python main.py visualize --model minilm
  python main.py info
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Reconfigure stdout and stderr on Windows to use UTF-8 and avoid encoding crashes with emojis
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from rich.columns import Columns
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table

# pyrefly: ignore [missing-import]
from embedder import MODELS, SemanticEmbedder
# pyrefly: ignore [missing-import]
from sentences import CATEGORIES, SENTENCES

# ── App & console setup ──────────────────────────────────────────────────────────
app     = typer.Typer(help="Semantic Search CLI — CalderR Lab 3.1", add_completion=False)
console = Console()

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────────
def cache_path(model_key: str) -> Path:
    return CACHE_DIR / f"embeddings_{model_key}.npy"


def load_or_embed(model_key: str, show_info: bool = True) -> tuple[SemanticEmbedder, np.ndarray]:
    """Return (embedder, corpus_embeddings), loading from cache if available."""
    path = cache_path(model_key)
    embedder = SemanticEmbedder(model_key)

    if path.exists():
        if show_info:
            console.print(f"  [dim]↑ loaded cached embeddings for [bold]{MODELS[model_key]}[/bold][/dim]")
        embeddings = np.load(path)
    else:
        console.print(f"  [yellow]Embedding {len(SENTENCES)} sentences with [bold]{MODELS[model_key]}[/bold]…[/yellow]")
        embeddings = embedder.embed(SENTENCES, show_progress=True)
        np.save(path, embeddings)
        console.print(f"  [green]✓ cached → {path}[/green]")

    return embedder, embeddings


def results_table(
    results: list[tuple[str, float]],
    title: str,
    border: str = "blue",
) -> Table:
    table = Table(title=title, border_style=border, show_lines=True, expand=True)
    table.add_column("#",      style="cyan",  width=4,  no_wrap=True)
    table.add_column("Score",  style="green", width=8,  no_wrap=True)
    table.add_column("Sentence", style="white")

    for rank, (sentence, score) in enumerate(results, 1):
        # Colour-code by score bracket
        score_colour = (
            "bright_green" if score > 0.70
            else "green"   if score > 0.50
            else "yellow"  if score > 0.30
            else "red"
        )
        table.add_row(str(rank), f"[{score_colour}]{score:.4f}[/{score_colour}]", sentence)

    return table


# ── Commands ─────────────────────────────────────────────────────────────────────
@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query"),
    model: str = typer.Option("minilm", "--model", "-m",
                               help=f"Embedding model key: {', '.join(MODELS)}"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
) -> None:
    """Search the 100-sentence corpus using semantic similarity."""
    if top_k <= 0:
        console.print("[red]Error: --top-k must be greater than 0.[/red]")
        raise typer.Exit(1)

    if model not in MODELS:
        console.print(f"[red]Unknown model '{model}'. Use: {', '.join(MODELS)}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold]  {query}\n[dim]Model:   {MODELS[model]}\nTop-k:   {top_k}[/dim]",
        title="🔍 Semantic Search", border_style="blue"
    ))

    embedder, embeddings = load_or_embed(model)
    results, latency     = embedder.search(query, embeddings, SENTENCES, top_k)

    console.print()
    console.print(results_table(results, title=f"Results  [{latency * 1000:.0f} ms]"))
    console.print()


@app.command()
def compare(
    query: str = typer.Argument(..., help="Natural-language search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results per model"),
) -> None:
    """
    Compare search results between all-MiniLM-L6-v2 and bge-small-en side-by-side.

    Prints two result tables and an agreement summary showing how many
    sentences both models agree are relevant.
    """
    if top_k <= 0:
        console.print("[red]Error: --top-k must be greater than 0.[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold]  {query}\n[dim]Comparing: {' vs '.join(MODELS.values())}[/dim]",
        title="⚖  Model Comparison", border_style="magenta"
    ))
    console.print()

    all_results: dict[str, tuple[list, float]] = {}
    for key in MODELS:
        console.print(f"[dim]Loading {key}…[/dim]")
        embedder, embeddings = load_or_embed(key, show_info=False)
        results, latency     = embedder.search(query, embeddings, SENTENCES, top_k)
        all_results[key]     = (results, latency)

    # ── Side-by-side tables ───────────────────────────────────────────────────
    colours = {"minilm": "blue", "bge": "magenta"}
    tables  = []
    result_sets: dict[str, set[str]] = {}

    for key, (results, latency) in all_results.items():
        t = results_table(
            results,
            title=f"{MODELS[key]}\n[dim][{latency * 1000:.0f} ms][/dim]",
            border=colours[key],
        )
        tables.append(t)
        result_sets[key] = {s for s, _ in results}

    # Print tables side-by-side if terminal is wide enough
    console.print(Columns(tables, equal=True, expand=True))
    console.print()

    # ── Agreement analysis ────────────────────────────────────────────────────
    overlap   = result_sets["minilm"] & result_sets["bge"]
    only_mini = result_sets["minilm"] - result_sets["bge"]
    only_bge  = result_sets["bge"]  - result_sets["minilm"]

    agreement = len(overlap) / top_k * 100
    colour    = "green" if agreement >= 60 else "yellow" if agreement >= 40 else "red"

    summary = (
        f"[bold]Agreement:[/bold]  [{colour}]{len(overlap)}/{top_k} ({agreement:.0f}%)[/{colour}]\n"
        f"Both agree:  {len(overlap)} sentence(s)\n"
        f"MiniLM only: {len(only_mini)} | BGE only: {len(only_bge)}"
    )
    console.print(Panel(summary, title="📊 Model Agreement", border_style="yellow"))
    console.print()


@app.command()
def visualize(
    model: str = typer.Option("minilm", "--model", "-m",
                               help="Model to visualise embeddings for"),
    output: str = typer.Option("embeddings_pca.png", "--output", "-o",
                                help="Output image filename"),
) -> None:
    """
    Project 100 sentence embeddings to 2D using PCA and save a scatter plot.

    Points are colour-coded by category so you can see which topics cluster
    together in the embedding space.  Run after first `search` to reuse cache.
    """
    try:
        # pyrefly: ignore [missing-import]
        import matplotlib.pyplot as plt
        # pyrefly: ignore [missing-import]
        from sklearn.decomposition import PCA
    except ImportError:
        console.print("[red]Install extras:  pip install matplotlib scikit-learn[/red]")
        raise typer.Exit(1)

    console.print()
    _, embeddings = load_or_embed(model)

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(embeddings)  # (100, 2)
    var_pc1 = pca.explained_variance_ratio_[0]
    var_pc2 = pca.explained_variance_ratio_[1]

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        # pyrefly: ignore [missing-import]
        import matplotlib
        cmap = matplotlib.colormaps["tab20"]
    except (ImportError, AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20", len(CATEGORIES))
    fig, ax = plt.subplots(figsize=(16, 11))

    for i, category in enumerate(CATEGORIES):
        idx    = slice(i * 5, (i + 1) * 5)
        colour = cmap(i)
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=[colour], label=category, alpha=0.85, s=80, edgecolors="white", linewidths=0.4,
        )

    ax.set_title(
        f"Sentence Embeddings — PCA 2D Projection\nModel: {MODELS[model]}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel(f"PC1  ({var_pc1:.1%} variance explained)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var_pc2:.1%} variance explained)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    fig.savefig(output, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ PCA plot saved → {output}[/green]")
    console.print(f"  Total variance explained: {(var_pc1 + var_pc2):.1%}")
    console.print()


@app.command()
def info() -> None:
    """Show corpus statistics and cache status."""
    console.print()

    # ── Corpus summary ─────────────────────────────────────────────────────────
    t = Table(title="Corpus", border_style="cyan", show_lines=False)
    t.add_column("Metric",  style="cyan")
    t.add_column("Value",   style="white")
    t.add_row("Total sentences",  str(len(SENTENCES)))
    t.add_row("Categories",       str(len(CATEGORIES)))
    t.add_row("Sentences/category", "5")
    console.print(t)
    console.print()

    # ── Model & cache info ─────────────────────────────────────────────────────
    t2 = Table(title="Models & Cache", border_style="blue", show_lines=False)
    t2.add_column("Key",         style="cyan")
    t2.add_column("Model name",  style="white")
    t2.add_column("Dimensions",  style="dim")
    t2.add_column("Cache",       style="green")

    dims = {"minilm": "384", "bge": "384"}
    for key, name in MODELS.items():
        cp     = cache_path(key)
        cached = "[green]✓ cached[/green]" if cp.exists() else "[yellow]not cached[/yellow]"
        t2.add_row(key, name, dims[key], cached)

    console.print(t2)
    console.print()


# ── Entry point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
