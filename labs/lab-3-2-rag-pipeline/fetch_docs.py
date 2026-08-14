"""
fetch_docs.py — Step 1: Download Wikipedia articles and save to docs/

Fetches 10 articles on AI/ML topics, each roughly 3–6 pages of text.
Saves two files per article:
  docs/{slug}.txt   — the full article text
  docs/{slug}.json  — metadata sidecar (title, url, word_count, fetch_date)

Run once before ingesting:
  python fetch_docs.py
  python fetch_docs.py --list      # show what's already downloaded
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.table import Table

# Reconfigure stdout to support UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

app     = typer.Typer(help="Fetch Wikipedia articles for Lab 3.2", add_completion=False)
console = Console()

DOCS_DIR = Path("./docs")

# 10 articles — diverse enough that RAG queries will be interesting
TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Large language model",
    "Transformer (deep learning architecture)",
    "Vector database",
    "Retrieval-augmented generation",
    "Reinforcement learning",
    "Computer vision",
]


def slugify(title: str) -> str:
    """Convert article title to a safe filename."""
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_")


def fetch_article(topic: str) -> dict | None:
    """
    Download a single Wikipedia article.
    Returns a dict with keys: title, url, content, word_count, slug
    Returns None on failure.
    """
    try:
        # pyrefly: ignore [missing-import]
        import wikipedia  # pip install wikipedia
    except ImportError:
        console.print("[red]Missing package: pip install wikipedia[/red]")
        sys.exit(1)

    wikipedia.set_lang("en")
    wikipedia.set_user_agent("CalderrRAGPipeline/1.0 (adeel@example.com)")

    try:
        page    = wikipedia.page(topic, auto_suggest=False)
        content = page.content

        # Light clean-up: remove section markers (== Heading ==) but keep text
        content = re.sub(r"={2,}.*?={2,}", "", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        return {
            "title":      page.title,
            "url":        page.url,
            "content":    content,
            "word_count": len(content.split()),
            "slug":       slugify(page.title),
            "fetch_date": datetime.now(timezone.utc).isoformat(),
        }

    except wikipedia.DisambiguationError as e:
        # Pick the first option that looks right
        console.print(f"  [yellow]Disambiguation for '{topic}' — trying first option[/yellow]")
        try:
            page    = wikipedia.page(e.options[0], auto_suggest=False)
            content = page.content
            content = re.sub(r"={2,}.*?={2,}", "", content)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            return {
                "title":      page.title,
                "url":        page.url,
                "content":    content,
                "word_count": len(content.split()),
                "slug":       slugify(page.title),
                "fetch_date": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            return None

    except Exception as e:
        console.print(f"  [red]Failed: {e}[/red]")
        return None


@app.command()
def fetch(
    force: bool = typer.Option(False, "--force", "-f",
                                help="Re-download even if file already exists"),
) -> None:
    """Download all 10 Wikipedia articles to docs/."""
    DOCS_DIR.mkdir(exist_ok=True)

    console.print(f"\n[bold]Fetching {len(TOPICS)} Wikipedia articles → {DOCS_DIR}/[/bold]\n")

    results = []
    for topic in TOPICS:
        txt_path = None  # will be set if slug available

        # Check cache first
        slug_guess = slugify(topic)
        cached_txt = DOCS_DIR / f"{slug_guess}.txt"
        if cached_txt.exists() and not force:
            meta_path = cached_txt.with_suffix(".json")
            meta      = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            wc        = meta.get("word_count", "?")
            console.print(f"  [dim]↑ already downloaded:[/dim] [cyan]{cached_txt.name}[/cyan] ({wc} words)")
            results.append({"topic": topic, "status": "cached", "words": wc})
            continue

        console.print(f"  Fetching: [yellow]{topic}[/yellow]")
        article = fetch_article(topic)

        if article is None:
            console.print(f"  [red]✗ failed[/red]")
            results.append({"topic": topic, "status": "failed", "words": 0})
            continue

        slug     = article["slug"]
        txt_path = DOCS_DIR / f"{slug}.txt"
        meta_path = DOCS_DIR / f"{slug}.json"

        txt_path.write_text(article["content"], encoding="utf-8")
        meta_path.write_text(json.dumps({
            "title":      article["title"],
            "url":        article["url"],
            "word_count": article["word_count"],
            "fetch_date": article["fetch_date"],
            "slug":       slug,
        }, indent=2), encoding="utf-8")

        est_pages = article["word_count"] // 300  # ~300 words/page
        console.print(f"  [green]✓[/green] {txt_path.name}  ({article['word_count']:,} words ≈ {est_pages} pages)")
        results.append({"topic": topic, "status": "ok", "words": article["word_count"]})

    # Summary table
    console.print()
    t = Table(title="Fetch Summary", border_style="green", show_lines=False)
    t.add_column("Topic",       style="cyan")
    t.add_column("Status",      style="white")
    t.add_column("Words",       style="green", justify="right")
    t.add_column("Est. pages",  style="dim",   justify="right")

    total_words = 0
    for r in results:
        status_str = (
            "[green]✓ ok[/green]"     if r["status"] == "ok"
            else "[dim]cached[/dim]"  if r["status"] == "cached"
            else "[red]✗ failed[/red]"
        )
        w = r["words"] if isinstance(r["words"], int) else 0
        total_words += w
        t.add_row(r["topic"], status_str, f"{w:,}", str(w // 300))

    t.add_section()
    t.add_row("[bold]TOTAL[/bold]", "", f"[bold]{total_words:,}[/bold]", f"[bold]{total_words // 300}[/bold]")
    console.print(t)
    console.print()


@app.command()
def list_docs() -> None:
    """Show all downloaded documents in docs/."""
    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        console.print("[yellow]No documents yet. Run: python fetch_docs.py fetch[/yellow]")
        return

    t = Table(title=f"Documents in {DOCS_DIR}/", border_style="cyan", show_lines=False)
    t.add_column("File",        style="cyan")
    t.add_column("Words",       style="green",  justify="right")
    t.add_column("Fetched",     style="dim")
    t.add_column("URL",         style="dim")

    total = 0
    for txt in files:
        meta_path = txt.with_suffix(".json")
        if meta_path.exists():
            m  = json.loads(meta_path.read_text())
            wc = m.get("word_count", "?")
            dt = m.get("fetch_date", "")[:10]
            url = m.get("url", "")
        else:
            wc  = len(txt.read_text().split())
            dt  = "unknown"
            url = ""
        total += wc if isinstance(wc, int) else 0
        t.add_row(txt.name, f"{wc:,}", dt, url[:60])

    t.add_section()
    t.add_row("[bold]TOTAL[/bold]", f"[bold]{total:,}[/bold]", "", "")
    console.print(t)
    console.print()


if __name__ == "__main__":
    # Default to "fetch" if no arguments provided, otherwise let Typer parse argv
    if len(sys.argv) > 1:
        app()
    else:
        app(["fetch"])
