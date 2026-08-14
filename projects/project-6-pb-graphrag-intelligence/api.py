"""
api.py — FastAPI Control Plane for Project 6-PB GraphRAG Knowledge Intelligence.

Endpoints:
- POST /evaluate: Evaluates query with selectable query mode (auto, vector_only, graph_only, hybrid).
- GET /questions: Returns verified 30-question benchmark dataset.
- GET /evaluation-report: Returns evaluation summary with explicit mode ("stub" or "real").
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException

PROJ_DIR = Path(__file__).resolve().parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

try:
    # pyrefly: ignore [missing-import]
    from .api_models import EvaluationReportSummaryResponse, QueryModeRequest, QueryModeResponse
    # pyrefly: ignore [missing-import]
    from .dataset import get_verified_benchmark_dataset
    # pyrefly: ignore [missing-import]
    from .evaluator import EvaluationRunner
    # pyrefly: ignore [missing-import]
    from .hybrid_retriever import GraphRAGHybridRetriever
    # pyrefly: ignore [missing-import]
    from .models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from .router import QueryRouter
except ImportError:
    # pyrefly: ignore [missing-import]
    from api_models import EvaluationReportSummaryResponse, QueryModeRequest, QueryModeResponse
    # pyrefly: ignore [missing-import]
    from dataset import get_verified_benchmark_dataset
    # pyrefly: ignore [missing-import]
    from evaluator import EvaluationRunner
    # pyrefly: ignore [missing-import]
    from hybrid_retriever import GraphRAGHybridRetriever
    # pyrefly: ignore [missing-import]
    from models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from router import QueryRouter

app = FastAPI(
    title="Project 6-P-B: GraphRAG Knowledge Intelligence API",
    description="FastAPI control plane for dual-indexed GraphRAG hybrid retrieval and evaluation.",
    version="1.0.0",
)

# Singleton instances for API execution
retriever = GraphRAGHybridRetriever(use_real=False)
router = QueryRouter(use_real=False)


@app.get("/")
def root() -> dict[str, str]:
    """Root status endpoint."""
    return {
        "status": "online",
        "service": "Project 6-P-B GraphRAG Knowledge Intelligence API",
        "docs": "/docs",
    }


@app.post("/evaluate", response_model=QueryModeResponse)
def evaluate_query(req: QueryModeRequest) -> QueryModeResponse:
    """
    Evaluate natural language query using selectable query mode.

    Modes:
    - 'auto': pre-retrieval QueryRouter classifies question into factual/relational/complex.
    - 'vector_only': forces vector search over ChromaDB index.
    - 'graph_only': forces NetworkX graph neighborhood expansion.
    - 'hybrid': forces combined vector + graph search with context deduplication.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question string cannot be empty.")

    if req.mode == "auto":
        cat = router.classify(req.question)
        cat_map = {"factual": "vector_only", "relational": "graph_only", "complex": "hybrid"}
        mode_used = cat_map.get(cat, "hybrid")
    else:
        mode_used = req.mode  # type: ignore[assignment]

    paras, context_str = retriever.retrieve_context(req.question, method=mode_used)
    answer = retriever.generate_answer(req.question, context_str)

    return QueryModeResponse(
        question=req.question,
        mode_used=mode_used,
        answer=answer,
        context_used=context_str,
    )


@app.get("/questions", response_model=List[QuestionRecord])
def get_benchmark_questions() -> List[QuestionRecord]:
    """Return the verified 30-question benchmark dataset (10 Factual, 10 Relational, 10 Complex)."""
    return get_verified_benchmark_dataset()


@app.get("/evaluation-report", response_model=EvaluationReportSummaryResponse)
def get_evaluation_report() -> EvaluationReportSummaryResponse:
    """
    Return the 30-question evaluation summary.
    Includes explicit mode field ('stub' or 'real') to clearly distinguish synthetic verification from real LLM runs.
    """
    dataset = get_verified_benchmark_dataset()
    runner = EvaluationRunner(use_real=False)
    records = runner.run_evaluation(dataset)

    summary_map: dict[str, dict[str, float]] = {}
    categories = ["factual", "relational", "complex"]
    methods = ["vector_only", "graph_only", "hybrid"]

    for cat in categories:
        summary_map[cat] = {}
        for m in methods:
            recs = [r for r in records if r.category == cat and r.method == m]
            if recs:
                avg = sum((r.faithfulness + r.response_relevancy + r.context_precision + r.context_recall) / 4.0 for r in recs) / len(recs)
                summary_map[cat][m] = round(avg, 3)
            else:
                summary_map[cat][m] = 0.0

    return EvaluationReportSummaryResponse(
        mode="stub",
        sample_size=30,
        summary=summary_map,
    )
