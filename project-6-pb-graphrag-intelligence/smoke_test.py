"""
smoke_test.py — Validation suite for Project 6-PB GraphRAG Intelligence (Phases 1 & 2).

Executes Phase 1 proofs + 4 mandatory Phase 2 proofs:
1. Statistics-Correctness Proof (both directions: positive shift -> True, zero shift -> False).
2. Pipeline-Wiring Proof: 90 EvaluationRecord instances generated across 30 questions x 3 methods.
3. Ambiguity-Verification Proof: All 30 benchmark questions ground unambiguously to single graph paths.
4. Report Proof: evaluation_report.html generated and verified on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

try:
    from .dataset import BENCHMARK_30_QUESTIONS, get_verified_benchmark_dataset, verify_unambiguous
    from .eval_models import EvaluationRecord
    from .evaluator import EvaluationRunner
    from .graph_retrieval import GraphRetriever, build_knowledge_graph
    from .hybrid_retriever import GraphRAGHybridRetriever
    from .models import QuestionRecord
    from .report_generator import generate_html_report
    from .statistical_test import run_paired_ttest
except ImportError:
    from dataset import BENCHMARK_30_QUESTIONS, get_verified_benchmark_dataset, verify_unambiguous
    from eval_models import EvaluationRecord
    from evaluator import EvaluationRunner
    from graph_retrieval import GraphRetriever, build_knowledge_graph
    from hybrid_retriever import GraphRAGHybridRetriever
    from models import QuestionRecord
    from report_generator import generate_html_report
    from statistical_test import run_paired_ttest


# ------------------------------------------------------------------------------
# Phase 1 Test Proofs
# ------------------------------------------------------------------------------
def test_scale_correctness() -> None:
    print("\n--- Test 1: Scale-Correctness Proof ---")
    graph = build_knowledge_graph()

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    print(f"Graph Node Count: {node_count} (Expected: 25)")
    print(f"Graph Edge Count: {edge_count} (Expected: 27)")

    assert node_count == 25, f"Scale correctness failed: expected 25 nodes, got {node_count}"
    assert edge_count == 27, f"Scale correctness failed: expected 27 edges, got {edge_count}"

    ridgeline_node = graph.nodes["Ridgeline Dynamics"]
    ridgeline_pids = ridgeline_node.get("source_paragraph_ids", [])
    print(f"Ridgeline Dynamics node source paragraph IDs: {ridgeline_pids}")
    assert len(ridgeline_pids) >= 3, "Deduplication failed: Ridgeline Dynamics should merge across multiple paragraphs"

    print("OK: Scale-Correctness Proof Passed!")


def test_sibling_expansion(retriever: GraphRAGHybridRetriever) -> None:
    print("\n--- Test 2: Sibling-Expansion Proof ---")

    q_meridian = "Besides Ridgeline Dynamics, what other company shares the same parent company?"
    _, entities_meridian = retriever.graph_retriever.retrieve(q_meridian)
    print(f"Meridian Holdings neighborhood entities: {entities_meridian}")

    assert "Ridgeline Dynamics" in entities_meridian, "Missing Ridgeline Dynamics in Meridian sibling expansion"
    assert "Kestrel Biotech" in entities_meridian, "Missing Kestrel Biotech in Meridian sibling expansion"
    assert "Meridian Holdings" in entities_meridian, "Missing Meridian Holdings parent in sibling expansion"

    q_atlas = "Besides Nimbus Water Systems, what other company shares the same parent company?"
    _, entities_atlas = retriever.graph_retriever.retrieve(q_atlas)
    print(f"Atlas Group neighborhood entities: {entities_atlas}")

    assert "Nimbus Water Systems" in entities_atlas, "Missing Nimbus Water Systems in Atlas sibling expansion"
    assert "Pinnacle Cargo Systems" in entities_atlas, "Missing Pinnacle Cargo Systems in Atlas sibling expansion"
    assert "Atlas Group" in entities_atlas, "Missing Atlas Group parent in sibling expansion"

    print("OK: Sibling-Expansion Proof Passed!")


# ------------------------------------------------------------------------------
# Phase 2 Mandatory Test Proofs
# ------------------------------------------------------------------------------
def test_statistics_correctness_both_directions() -> None:
    print("\n--- Test 3: Statistics-Correctness Proof (Both Directions) ---")

    # Case A: Positive Shift (Hybrid > Vector by +0.3 consistently across 10 complex questions)
    pos_records: list[EvaluationRecord] = []
    for i in range(10):
        q = f"Complex question {i+1}"
        pos_records.append(
            EvaluationRecord(
                question=q,
                category="complex",
                method="hybrid",
                faithfulness=0.95,
                response_relevancy=0.95,
                context_precision=0.95,
                context_recall=0.95,
            )
        )
        pos_records.append(
            EvaluationRecord(
                question=q,
                category="complex",
                method="vector_only",
                faithfulness=0.60,
                response_relevancy=0.60,
                context_precision=0.60,
                context_recall=0.60,
            )
        )

    sig_pos = run_paired_ttest(pos_records, category="complex")
    print(f"Case A (Positive Shift) -> t-stat: {sig_pos.t_statistic:.4f}, p-val: {sig_pos.p_value:.6f}, significant_at_05: {sig_pos.significant_at_05}")
    assert sig_pos.significant_at_05 is True, "Statistics test failed: positive shift must be statistically significant (True)"

    # Case B: Zero Shift (Hybrid == Vector identical scores across 10 complex questions)
    zero_records: list[EvaluationRecord] = []
    for i in range(10):
        q = f"Complex question {i+1}"
        zero_records.append(
            EvaluationRecord(
                question=q,
                category="complex",
                method="hybrid",
                faithfulness=0.80,
                response_relevancy=0.80,
                context_precision=0.80,
                context_recall=0.80,
            )
        )
        zero_records.append(
            EvaluationRecord(
                question=q,
                category="complex",
                method="vector_only",
                faithfulness=0.80,
                response_relevancy=0.80,
                context_precision=0.80,
                context_recall=0.80,
            )
        )

    sig_zero = run_paired_ttest(zero_records, category="complex")
    print(f"Case B (Zero Shift) -> t-stat: {sig_zero.t_statistic:.4f}, p-val: {sig_zero.p_value:.6f}, significant_at_05: {sig_zero.significant_at_05}")
    assert sig_zero.significant_at_05 is False, "Statistics test failed: zero shift must NOT be statistically significant (False)"

    print("OK: Statistics-Correctness Proof Passed!")


def test_pipeline_wiring() -> None:
    print("\n--- Test 4: Pipeline-Wiring Proof ---")
    dataset = get_verified_benchmark_dataset()

    assert len(dataset) == 30, f"Expected 30 benchmark questions, got {len(dataset)}"

    runner = EvaluationRunner(use_real=False)
    records = runner.run_evaluation(dataset)

    print(f"Total Evaluation Records Produced: {len(records)} (Expected: 90)")
    assert len(records) == 90, f"Expected 90 evaluation records (30x3), got {len(records)}"

    factual_recs = [r for r in records if r.category == "factual"]
    relational_recs = [r for r in records if r.category == "relational"]
    complex_recs = [r for r in records if r.category == "complex"]

    print(f"Category Distribution -> Factual: {len(factual_recs)}/30 | Relational: {len(relational_recs)}/30 | Complex: {len(complex_recs)}/30")

    assert len(factual_recs) == 30, "Factual category records mismatch"
    assert len(relational_recs) == 30, "Relational category records mismatch"
    assert len(complex_recs) == 30, "Complex category records mismatch"

    print("OK: Pipeline-Wiring Proof Passed!")


def test_ambiguity_verification() -> None:
    print("\n--- Test 5: Ambiguity-Verification Proof ---")
    graph_retriever = GraphRetriever()

    unambiguous_count = 0
    for q in BENCHMARK_30_QUESTIONS:
        is_unambiguous = verify_unambiguous(q, graph_retriever)
        if is_unambiguous:
            unambiguous_count += 1

    print(f"Unambiguous Questions Verified: {unambiguous_count}/30")
    assert unambiguous_count == 30, f"Ambiguity verification failed: only {unambiguous_count}/30 questions passed"

    print("OK: Ambiguity-Verification Proof Passed!")


def test_report_generation() -> None:
    print("\n--- Test 6: HTML Report Proof ---")
    dataset = get_verified_benchmark_dataset()
    runner = EvaluationRunner(use_real=False)
    records = runner.run_evaluation(dataset)
    sig_result = run_paired_ttest(records, category="complex")

    report_path = PROJ_DIR / "evaluation_report.html"
    generated_path = generate_html_report(records, sig_result, is_real=False, output_path=report_path)

    print(f"Generated Report Path: {generated_path}")
    assert generated_path.exists(), "Report file does not exist on disk"
    assert generated_path.stat().st_size > 0, "Report file is empty"

    with open(generated_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Project 6-PB: GraphRAG 30-Question Evaluation Study" in content, "Missing header in HTML report"
    assert "Paired t-Test Statistical Significance" in content, "Missing statistical test panel in HTML report"

    print("OK: HTML Report Proof Passed!")


def main() -> None:
    print("================================================================================")
    print("      PROJECT 6-PB PHASE 2: GRAPH RAG EVALUATION - SMOKE TEST SUITE             ")
    print("================================================================================")

    retriever = GraphRAGHybridRetriever(use_real=False)

    test_scale_correctness()
    test_sibling_expansion(retriever)
    test_statistics_correctness_both_directions()
    test_pipeline_wiring()
    test_ambiguity_verification()
    test_report_generation()

    print("\n================================================================================")
    print("ALL PHASE 1 AND PHASE 2 SMOKE TEST SUITE PROOFS PASSED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
