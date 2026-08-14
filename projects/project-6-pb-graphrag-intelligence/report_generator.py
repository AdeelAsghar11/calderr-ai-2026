"""
report_generator.py — HTML Evaluation Report Generator for Project 6-PB Phase 2.

Exports evaluation_report.html containing per-method RAGAS scores for all 30 questions
and the paired t-test statistical significance summary panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

try:
    from .eval_models import EvaluationRecord, SignificanceResult
except ImportError:
    from eval_models import EvaluationRecord, SignificanceResult


def generate_html_report(
    records: List[EvaluationRecord],
    sig_result: SignificanceResult,
    is_real: bool = False,
    output_path: Path | None = None,
) -> Path:
    """
    Generate evaluation_report.html formatted report.

    Args:
        records: All 90 EvaluationRecord instances.
        sig_result: Statistical significance test output.
        is_real: True if generated using ChatGroq real RAGAS scoring; False if stub mode.
        output_path: Target HTML file path.

    Returns:
        Path to generated HTML report.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "evaluation_report.html"

    mode_label = "REAL RAGAS EVALUATION (ChatGroq llama-3.3-70b-versatile)" if is_real else "STUB MODE SIMULATION (Plumbing Verification Only)"
    mode_color = "#2e7d32" if is_real else "#d32f2f"

    # Group records by question
    question_map: dict[str, dict[str, EvaluationRecord]] = {}
    question_cats: dict[str, str] = {}

    for r in records:
        q = r.question
        if q not in question_map:
            question_map[q] = {}
        question_map[q][r.method] = r
        question_cats[q] = r.category

    rows_html = ""
    for i, (q, m_map) in enumerate(question_map.items(), 1):
        cat = question_cats[q]
        v_rec = m_map.get("vector_only")
        g_rec = m_map.get("graph_only")
        h_rec = m_map.get("hybrid")

        v_score = f"{(v_rec.faithfulness + v_rec.response_relevancy + v_rec.context_precision + v_rec.context_recall)/4.0:.3f}" if v_rec else "N/A"
        g_score = f"{(g_rec.faithfulness + g_rec.response_relevancy + g_rec.context_precision + g_rec.context_recall)/4.0:.3f}" if g_rec else "N/A"
        h_score = f"{(h_rec.faithfulness + h_rec.response_relevancy + h_rec.context_precision + h_rec.context_recall)/4.0:.3f}" if h_rec else "N/A"

        rows_html += f"""
        <tr>
            <td>{i}</td>
            <td>{q}</td>
            <td><span class="badge {cat}">{cat.upper()}</span></td>
            <td>{v_score}</td>
            <td>{g_score}</td>
            <td><strong>{h_score}</strong></td>
        </tr>
        """

    sig_status_text = "STATISTICALLY SIGNIFICANT (p < 0.05)" if sig_result.significant_at_05 else "NOT SIGNIFICANT (p >= 0.05)"
    sig_status_color = "#2e7d32" if sig_result.significant_at_05 else "#c62828"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Project 6-PB GraphRAG Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
        .header-banner {{ background-color: {mode_color}; color: white; padding: 12px 20px; border-radius: 6px; font-weight: bold; margin-bottom: 25px; }}
        .summary-card {{ background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .metric-grid {{ display: flex; gap: 20px; margin-top: 15px; }}
        .metric-box {{ background-color: #f0f4f8; padding: 15px; border-radius: 6px; flex: 1; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #0d47a1; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background-color: #1a237e; color: white; font-weight: 600; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white; }}
        .badge.factual {{ background-color: #0288d1; }}
        .badge.relational {{ background-color: #7b1fa2; }}
        .badge.complex {{ background-color: #e65100; }}
    </style>
</head>
<body>

    <h1>Project 6-PB: GraphRAG 30-Question Evaluation Study</h1>
    <div class="header-banner">Mode: {mode_label}</div>

    <div class="summary-card">
        <h2>Paired t-Test Statistical Significance (Complex Category)</h2>
        <p>Comparing Hybrid Retrieval vs. Vector-Only Retrieval on 10 Complex Questions:</p>
        <div class="metric-grid">
            <div class="metric-box">
                <div>Sample Size (n)</div>
                <div class="metric-value">{sig_result.sample_size}</div>
            </div>
            <div class="metric-box">
                <div>t-Statistic</div>
                <div class="metric-value">{sig_result.t_statistic:.4f}</div>
            </div>
            <div class="metric-box">
                <div>p-Value</div>
                <div class="metric-value">{sig_result.p_value:.6f}</div>
            </div>
            <div class="metric-box" style="background-color: {sig_status_color}; color: white;">
                <div>Significance (p &lt; 0.05)</div>
                <div class="metric-value" style="color: white;">{sig_status_text}</div>
            </div>
        </div>
    </div>

    <h2>Full 30-Question RAGAS Benchmark Scores</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Question</th>
                <th>Category</th>
                <th>Vector-Only Mean</th>
                <th>Graph-Only Mean</th>
                <th>Hybrid Mean</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
