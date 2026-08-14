"""
eval_models.py — Strongly typed Pydantic data models for Phase 2 evaluation & statistical significance.

Defines schemas for EvaluationRecord (per-sample scores) and SignificanceResult (paired t-test output).
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class EvaluationRecord(BaseModel):
    """Evaluation record capturing per-sample RAGAS metrics for a question-method pair."""

    question: str = Field(description="The natural language query evaluated.")
    category: Literal["factual", "relational", "complex"] = Field(
        description="Ground truth question category."
    )
    method: Literal["vector_only", "graph_only", "hybrid"] = Field(
        description="Retrieval method executed."
    )
    faithfulness: float = Field(
        description="RAGAS Faithfulness score measuring factual alignment with context [0.0, 1.0]."
    )
    response_relevancy: float = Field(
        description="RAGAS Response Relevancy score measuring answer directness [0.0, 1.0]."
    )
    context_precision: float = Field(
        description="RAGAS Context Precision score measuring signal-to-noise ratio in retrieved context [0.0, 1.0]."
    )
    context_recall: float = Field(
        description="RAGAS Context Recall score measuring retrieval coverage of target ground truth [0.0, 1.0]."
    )


class SignificanceResult(BaseModel):
    """Statistical significance result from paired t-test comparing hybrid vs vector-only."""

    category: str = Field(description="Question category evaluated (e.g. 'complex').")
    metric_compared: str = Field(
        description="Metric compared between methods (e.g. 'mean_ragas_score')."
    )
    t_statistic: float = Field(description="Calculated paired t-test t-statistic.")
    p_value: float = Field(description="Calculated p-value.")
    significant_at_05: bool = Field(
        description="True if p_value < 0.05, indicating statistically significant difference."
    )
    sample_size: int = Field(description="Number of paired question samples evaluated (e.g. 10).")
