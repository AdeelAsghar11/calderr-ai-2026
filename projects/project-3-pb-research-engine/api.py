"""
api.py — FastAPI Backend

The REST API layer. Exposes the research engine as HTTP endpoints
so it can be called from the Streamlit UI, Postman, curl, or any client.

Endpoints:
  POST /research          — main endpoint, takes query, returns report
  GET  /reports           — list all saved reports
  GET  /reports/{id}      — get a specific saved report
  GET  /health            — health check (confirms all components loaded)

Why FastAPI:
  - Auto-generates OpenAPI docs at /docs (visit http://localhost:8000/docs)
  - Pydantic request/response validation built in
  - Async support means I/O-bound work (LLM calls) doesn't block other requests
  - Background tasks for saving reports to disk

How reports are saved:
  Every research request saves a JSON file to reports/{report_id}.json.
  This gives you the "10 research reports" deliverable automatically —
  just run 10 queries and the reports are saved.

Run:
  python api.py
  # or
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# pyrefly: ignore [missing-import]
import uvicorn
# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, BackgroundTasks
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

# pyrefly: ignore [missing-import]
from router import KnowledgeRouter
# pyrefly: ignore [missing-import]
from retriever import DualRetriever
# pyrefly: ignore [missing-import]
from report_generator import ReportGenerator, ResearchReport

load_dotenv(find_dotenv())

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Real-Time Research Engine",
    description = "Combines local knowledge base with live web search to generate cited research reports.",
    version     = "1.0.0",
)

# Allow Streamlit (running on 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Lazy-loaded components ────────────────────────────────────────────────────
# Load once on first request, not at startup — avoids slow boot times
_router    = None
_retriever = None
_generator = None


def get_components():
    global _router, _retriever, _generator
    if _router is None:
        _router    = KnowledgeRouter()
        _retriever = DualRetriever()
        _generator = ReportGenerator()
    return _router, _retriever, _generator


# ── Request / Response models ─────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query:       str
    source_mode: str = "auto"    # "auto" | "local" | "web" | "both"
    top_k:       int = 5


class ResearchResponse(BaseModel):
    report:           ResearchReport
    routing_decision: str
    routing_reason:   str
    latency_ms:       float


class ReportSummary(BaseModel):
    report_id:  str
    query:      str
    title:      str
    timestamp:  str
    sources:    list[str]


# ── Background task ───────────────────────────────────────────────────────────
def save_report(report: ResearchReport) -> None:
    """Save report to disk as JSON. Runs in background after response is sent."""
    path = REPORTS_DIR / f"{report.report_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Main research endpoint.

    Flow:
      1. Route the query (unless source_mode overrides)
      2. Retrieve from the selected source(s)
      3. Generate a cited research report
      4. Save the report in the background
      5. Return the report + metadata
    """
    import time
    t0 = time.perf_counter()

    router, retriever, generator = get_components()

    # ── Step 1: Route ─────────────────────────────────────────────────────────
    if request.source_mode == "auto":
        decision = router.route(request.query)
        source   = decision.source.value
        reason   = decision.reasoning
    else:
        source   = request.source_mode
        reason   = f"Manual override: {source}"

    # ── Step 2: Retrieve ──────────────────────────────────────────────────────
    docs = retriever.retrieve(request.query, source=source, top_k=request.top_k)

    # ── Step 3: Generate ──────────────────────────────────────────────────────
    import uuid
    report_id = str(uuid.uuid4())[:8]
    report    = generator.generate(
        query            = request.query,
        docs             = docs,
        routing_decision = source,
        report_id        = report_id,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── Step 4: Save in background ────────────────────────────────────────────
    background_tasks.add_task(save_report, report)

    return ResearchResponse(
        report           = report,
        routing_decision = source,
        routing_reason   = reason,
        latency_ms       = latency_ms,
    )


@app.get("/reports", response_model=list[ReportSummary])
async def list_reports():
    """List all saved research reports."""
    summaries = []
    for path in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(ReportSummary(
                report_id = data.get("report_id", path.stem),
                query     = data.get("query",     ""),
                title     = data.get("title",     ""),
                timestamp = data.get("timestamp", ""),
                sources   = data.get("sources_used", []),
            ))
        except Exception:
            continue
    return summaries


@app.get("/reports/{report_id}", response_model=ResearchReport)
async def get_report(report_id: str):
    """Get a specific saved report by ID."""
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return ResearchReport.model_validate_json(path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Health check. Returns component status."""
    chroma_ok = Path("../labs/lab-3-2-rag-pipeline/chroma_db").exists()
    return {
        "status":      "ok",
        "chromadb":    "ok" if chroma_ok else "not found",
        "groq_key":    "set" if os.getenv("GROQ_API_KEY") else "missing",
        "reports_dir": str(REPORTS_DIR),
        "saved":       len(list(REPORTS_DIR.glob("*.json"))),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
