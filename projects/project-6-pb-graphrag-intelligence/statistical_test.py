"""
statistical_test.py — Paired t-test statistical significance testing using scipy.stats.ttest_rel.

Compares hybrid vs vector-only evaluation scores on complex questions to prove statistically significant advantage.
"""

from __future__ import annotations

from typing import List
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy import stats

try:
    # pyrefly: ignore [missing-import]
    from .eval_models import EvaluationRecord, SignificanceResult
except ImportError:
    # pyrefly: ignore [missing-import]
    from eval_models import EvaluationRecord, SignificanceResult


def calculate_mean_score(rec: EvaluationRecord) -> float:
    """Calculate composite average RAGAS score across all 4 metrics."""
    return (rec.faithfulness + rec.response_relevancy + rec.context_precision + rec.context_recall) / 4.0


def run_paired_ttest(
    records: List[EvaluationRecord],
    category: str = "complex",
) -> SignificanceResult:
    """
    Execute paired t-test comparing hybrid vs vector_only on questions in target category.

    Args:
        records: List of EvaluationRecord instances from evaluation run.
        category: Target category to test (default: 'complex').

    Returns:
        SignificanceResult with t_statistic, p_value, and significant_at_05 boolean.
    """
    cat_records = [r for r in records if r.category == category]

    # Map question -> method -> score
    question_map: dict[str, dict[str, float]] = {}
    for r in cat_records:
        q = r.question
        if q not in question_map:
            question_map[q] = {}
        question_map[q][r.method] = calculate_mean_score(r)

    hybrid_scores: List[float] = []
    vector_scores: List[float] = []

    for q, m_dict in question_map.items():
        if "hybrid" in m_dict and "vector_only" in m_dict:
            hybrid_scores.append(m_dict["hybrid"])
            vector_scores.append(m_dict["vector_only"])

    sample_size = len(hybrid_scores)
    if sample_size == 0:
        return SignificanceResult(
            category=category,
            metric_compared="mean_ragas_score",
            t_statistic=0.0,
            p_value=1.0,
            significant_at_05=False,
            sample_size=0,
        )

    # Perform paired t-test
    t_stat, p_val = stats.ttest_rel(hybrid_scores, vector_scores)

    # Handle edge case where scores are identical or NaN
    if np.isnan(t_stat) or np.isnan(p_val):
        t_stat = 0.0
        p_val = 1.0

    significant_at_05 = bool(p_val < 0.05 and t_stat > 0)

    return SignificanceResult(
        category=category,
        metric_compared="mean_ragas_score",
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_val), 6),
        significant_at_05=significant_at_05,
        sample_size=sample_size,
    )
