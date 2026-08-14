"""
models.py — Strongly typed Pydantic data models for Project 6-PB GraphRAG Intelligence.

Defines schemas for benchmark question records, query router decisions, and retrieval method results.
"""

from __future__ import annotations

from typing import List, Literal
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


class QuestionRecord(BaseModel):
    """Benchmark question definition with category label and target correctness keywords."""

    question: str = Field(description="The natural language query string.")
    category: Literal["factual", "relational", "complex"] = Field(
        description="Ground truth question category."
    )
    expected_answer_keywords: List[str] = Field(
        description="Keywords that must appear in retrieved context to verify retrieval correctness."
    )


class RouterDecision(BaseModel):
    """Decision output from the Query Router predicting question category."""

    question: str = Field(description="The input query evaluated.")
    predicted_category: Literal["factual", "relational", "complex"] = Field(
        description="Category predicted by the router."
    )
    actual_category: Literal["factual", "relational", "complex"] = Field(
        description="Ground truth category of the question."
    )
    correct: bool = Field(description="True if predicted category matches actual category.")


class MethodResult(BaseModel):
    """Result of running a query through a specific retrieval method (vector_only, graph_only, or hybrid)."""

    method: Literal["vector_only", "graph_only", "hybrid"] = Field(
        description="Retrieval method executed."
    )
    context_used: str = Field(
        description="Concatenated and deduplicated text context passed to answer generation."
    )
    answer: str = Field(description="Generated answer string.")
    is_correct: bool = Field(
        description="True if all expected_answer_keywords were present in the retrieved context_used."
    )
