"""
api_models.py — Strongly typed Pydantic models for Project 6-PB FastAPI Control Plane.

Defines schemas for QueryModeRequest, QueryModeResponse, and EvaluationReportSummaryResponse.
"""

from __future__ import annotations

from typing import Dict, Literal
from pydantic import BaseModel, Field


class QueryModeRequest(BaseModel):
    """Request payload for /evaluate endpoint allowing query mode selection."""

    question: str = Field(description="Natural language query string.")
    mode: Literal["auto", "vector_only", "graph_only", "hybrid"] = Field(
        default="auto",
        description="Query mode: 'auto' uses QueryRouter, others override router choice.",
    )


class QueryModeResponse(BaseModel):
    """Response payload for /evaluate endpoint."""

    question: str = Field(description="Evaluated question string.")
    mode_used: Literal["vector_only", "graph_only", "hybrid"] = Field(
        description="Retrieval method executed."
    )
    answer: str = Field(description="Generated answer string.")
    context_used: str = Field(description="Deduplicated context string passed to answer generation.")


class EvaluationReportSummaryResponse(BaseModel):
    """Response payload for /evaluation-report endpoint."""

    mode: Literal["stub", "real"] = Field(
        description="Explicit evaluation mode label: 'stub' (synthetic verification) or 'real' (RAGAS ChatGroq)."
    )
    sample_size: int = Field(description="Number of benchmark questions evaluated (e.g. 30).")
    summary: Dict[str, Dict[str, float]] = Field(
        description="Nested dictionary of mean scores per category and method."
    )
