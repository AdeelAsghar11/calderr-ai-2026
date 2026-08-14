"""
generate_demos.py — Generate 10 Demo Research Reports

Runs 10 preset queries through the full pipeline and saves reports to reports/.
This satisfies the deliverable: "10 research reports on different topics."

The queries are chosen to demonstrate all three routing modes:
  - local-heavy (AI/ML concepts)
  - web-heavy (current events and products)
  - both (concepts + current state)

Run:
  python generate_demos.py
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.table import Table

load_dotenv(find_dotenv())

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

console = Console()

# ── 10 demo queries covering all three routing modes ──────────────────────────
DEMO_QUERIES = [
    # Local-heavy: foundational AI/ML concepts covered in our knowledge base
    {
        "query":       "What is the transformer architecture and how does attention work?",
        "source_hint": "local",
    },
    {
        "query":       "Explain backpropagation and how neural networks learn",
        "source_hint": "local",
    },
    {
        "query":       "What is retrieval augmented generation and why does it matter?",
        "source_hint": "local",
    },

    # Web-heavy: current information not in our knowledge base
    {
        "query":       "What are the most significant AI breakthroughs in 2026?",
        "source_hint": "web",
    },
    {
        "query":       "What is the current state of AI regulation globally?",
        "source_hint": "web",
    },
    {
        "query":       "Which vector databases are most popular for production RAG in 2026?",
        "source_hint": "web",
    },

    # Both: needs foundational knowledge AND current context
    {
        "query":       "How does RAG compare to the latest fine-tuned models?",
        "source_hint": "both",
    },
    {
        "query":       "What are the limitations of current large language models?",
        "source_hint": "both",
    },
    {
        "query":       "How is reinforcement learning being applied in real products today?",
        "source_hint": "both",
    },
    {
        "query":       "What advances have been made in computer vision since 2024?",
        "source_hint": "both",
    },
]


def run_query(query: str, router, retriever, generator) -> dict:
    """Run one query through the full pipeline. Returns report dict."""
    # Route
    decision = router.route(query)
    source   = decision.source.value

    # Retrieve
    docs = retriever.retrieve(query, source=source, top_k=5)

    # Generate
    report_id = str(uuid.uuid4())[:8]
    report    = generator.generate(
        query            = query,
        docs             = docs,
        routing_decision = source,
        report_id        = report_id,
    )

    # Save
    path = REPORTS_DIR / f"{report_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return {
        "report_id":  report_id,
        "query":      query,
        "source":     source,
        "reasoning":  decision.reasoning[:60],
        "title":      report.title,
        "citations":  len(report.citations),
        "findings":   len(report.key_findings),
    }


if __name__ == "__main__":
    console.print()
    console.print("[bold blue]Generating 10 demo research reports...[/bold blue]\n")

    # pyrefly: ignore [missing-import]
    from router import KnowledgeRouter
    # pyrefly: ignore [missing-import]
    from retriever import DualRetriever
    # pyrefly: ignore [missing-import]
    from report_generator import ReportGenerator

    router    = KnowledgeRouter()
    retriever = DualRetriever()
    generator = ReportGenerator()

    results = []
    for i, item in enumerate(DEMO_QUERIES, 1):
        console.print(f"  [{i:02d}/10] {item['query'][:60]}...")
        t0 = time.perf_counter()
        try:
            result = run_query(item["query"], router, retriever, generator)
            result["latency_ms"] = (time.perf_counter() - t0) * 1000
            result["status"] = "ok"
            console.print(f"         [green]✓[/green] source={result['source']}  citations={result['citations']}  {result['latency_ms']:.0f}ms")
        except Exception as e:
            result = {"query": item["query"], "status": "error", "error": str(e)}
            console.print(f"         [red]✗ {e}[/red]")
        results.append(result)
        time.sleep(0.5)  # avoid rate limiting

    # Summary table
    console.print()
    t = Table(title="Demo Reports Generated", border_style="green", show_lines=False)
    t.add_column("#",        style="dim",   width=4)
    t.add_column("Source",   style="cyan",  width=8)
    t.add_column("Title",    style="white", min_width=35)
    t.add_column("Cites",    style="green", width=6, justify="right")
    t.add_column("Status",   style="white", width=10)

    for i, r in enumerate(results, 1):
        status = "[green]✓ ok[/green]" if r.get("status") == "ok" else "[red]✗ error[/red]"
        t.add_row(
            str(i),
            r.get("source", "?"),
            r.get("title", r.get("query", "?"))[:45],
            str(r.get("citations", "-")),
            status,
        )

    console.print(t)
    ok = sum(1 for r in results if r.get("status") == "ok")
    console.print(f"\n[green]✓ {ok}/10 reports saved to reports/[/green]\n")
