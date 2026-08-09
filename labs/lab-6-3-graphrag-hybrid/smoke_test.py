"""
smoke_test.py — Offline validation suite for Lab 6.3 GraphRAG Hybrid Retrieval.

Executes 4 mandatory proofs:
1. Router Accuracy Proof: Router correctly classifies >= 12 out of 15 questions.
2. Category Advantage Proof: Vector > Graph on Factual; Graph > Vector on Relational.
3. Hybrid Necessity Proof: Hybrid strictly outperforms both Vector-only and Graph-only on Complex questions.
4. Deduplication Proof: Exact duplicate paragraphs are deduplicated to single occurrences in merged context.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package directory is in sys.path
LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

try:
    # pyrefly: ignore [missing-import]
    from .hybrid_retriever import GraphRAGHybridRetriever
    # pyrefly: ignore [missing-import]
    from .models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from .router import QueryRouter
except ImportError:
    # pyrefly: ignore [missing-import]
    from hybrid_retriever import GraphRAGHybridRetriever
    # pyrefly: ignore [missing-import]
    from models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from router import QueryRouter

# The 15 Benchmark Questions
BENCHMARK_QUESTIONS: list[QuestionRecord] = [
    # Factual (1-5)
    QuestionRecord(
        question="What was Dana Voss's profession before founding her company?",
        category="factual",
        expected_answer_keywords=["mechanical engineer"],
    ),
    QuestionRecord(
        question="What does Trailmark Robotics specialize in?",
        category="factual",
        expected_answer_keywords=["warehouse automation robots"],
    ),
    QuestionRecord(
        question="What did Rina Achebe do before founding Glacier Analytics?",
        category="factual",
        expected_answer_keywords=["satellite data scientist"],
    ),
    QuestionRecord(
        question="What does Glacier Analytics focus on?",
        category="factual",
        expected_answer_keywords=["agricultural satellite imagery"],
    ),
    QuestionRecord(
        question="What does Cobalt Freight specialize in?",
        category="factual",
        expected_answer_keywords=["cross-border shipping logistics"],
    ),
    # Relational (6-10)
    QuestionRecord(
        question="Who founded the company that Owen Kessler works at?",
        category="relational",
        expected_answer_keywords=["Rina Achebe"],
    ),
    QuestionRecord(
        question="What city is Cobalt Freight's parent company headquartered in?",
        category="relational",
        expected_answer_keywords=["Seattle"],
    ),
    QuestionRecord(
        question="Besides Trailmark Robotics, what other company shares the same parent company?",
        category="relational",
        expected_answer_keywords=["Cobalt Freight"],
    ),
    QuestionRecord(
        question="What city is Dana Voss's company located in?",
        category="relational",
        expected_answer_keywords=["Austin"],
    ),
    QuestionRecord(
        question="What city does Dana Voss's company's parent company operate in?",
        category="relational",
        expected_answer_keywords=["Seattle"],
    ),
    # Complex (11-15)
    QuestionRecord(
        question="What was the prior profession of the founder of the company Owen Kessler works at?",
        category="complex",
        expected_answer_keywords=["satellite data scientist"],
    ),
    QuestionRecord(
        question="What does the company that Dana Voss's company is part of do?",
        category="complex",
        expected_answer_keywords=["holding company"],
    ),
    QuestionRecord(
        question="What was the prior profession of the person who works at the company headquartered in Toronto?",
        category="complex",
        expected_answer_keywords=["data engineer"],
    ),
    QuestionRecord(
        question="What does the company that shares a parent with Trailmark Robotics specialize in?",
        category="complex",
        expected_answer_keywords=["cross-border shipping logistics"],
    ),
    QuestionRecord(
        question="What was the profession background of the founder of the company headquartered in Austin?",
        category="complex",
        expected_answer_keywords=["mechanical engineer"],
    ),
]


def test_1_router_accuracy(router: QueryRouter) -> int:
    print("\n--- Test 1: Router Accuracy Proof ---")
    correct_count = 0
    for q in BENCHMARK_QUESTIONS:
        decision = router.route(q)
        status = "PASS" if decision.correct else "FAIL"
        print(f"[{status}] Q: '{q.question}' | True: {q.category} | Predicted: {decision.predicted_category}")
        if decision.correct:
            correct_count += 1

    print(f"Router Accuracy: {correct_count}/{len(BENCHMARK_QUESTIONS)} ({(correct_count/15)*100:.1f}%)")
    assert (
        correct_count >= 12
    ), f"Router accuracy requirement failed: expected >= 12, got {correct_count}"
    return correct_count


def test_2_category_advantage(retriever: GraphRAGHybridRetriever) -> None:
    print("\n--- Test 2: Category Advantage Proof ---")

    factual_questions = [q for q in BENCHMARK_QUESTIONS if q.category == "factual"]
    relational_questions = [q for q in BENCHMARK_QUESTIONS if q.category == "relational"]

    # Factual category evaluation
    vector_factual_correct = 0
    graph_factual_correct = 0
    for q in factual_questions:
        vec_res = retriever.process_question(q, override_method="vector_only")
        graph_res = retriever.process_question(q, override_method="graph_only")
        if vec_res.is_correct:
            vector_factual_correct += 1
        if graph_res.is_correct:
            graph_factual_correct += 1

    print(
        f"Factual Category - Vector Only: {vector_factual_correct}/{len(factual_questions)} | Graph Only: {graph_factual_correct}/{len(factual_questions)}"
    )

    # Relational category evaluation
    vector_relational_correct = 0
    graph_relational_correct = 0
    for q in relational_questions:
        vec_res = retriever.process_question(q, override_method="vector_only")
        graph_res = retriever.process_question(q, override_method="graph_only")
        if vec_res.is_correct:
            vector_relational_correct += 1
        if graph_res.is_correct:
            graph_relational_correct += 1

    print(
        f"Relational Category - Graph Only: {graph_relational_correct}/{len(relational_questions)} | Vector Only: {vector_relational_correct}/{len(relational_questions)}"
    )

    assert (
        vector_factual_correct > graph_factual_correct
    ), f"Factual category advantage failed: vector ({vector_factual_correct}) must be > graph ({graph_factual_correct})"
    assert (
        graph_relational_correct > vector_relational_correct
    ), f"Relational category advantage failed: graph ({graph_relational_correct}) must be > vector ({vector_relational_correct})"
    print("OK: Category Advantage Proof Passed!")


def test_3_hybrid_necessity(retriever: GraphRAGHybridRetriever) -> tuple[int, int, int]:
    print("\n--- Test 3: Hybrid Necessity Proof ---")
    complex_questions = [q for q in BENCHMARK_QUESTIONS if q.category == "complex"]

    vector_correct = 0
    graph_correct = 0
    hybrid_correct = 0

    for q in complex_questions:
        vec_res = retriever.process_question(q, override_method="vector_only")
        graph_res = retriever.process_question(q, override_method="graph_only")
        hybrid_res = retriever.process_question(q, override_method="hybrid")

        if vec_res.is_correct:
            vector_correct += 1
        if graph_res.is_correct:
            graph_correct += 1
        if hybrid_res.is_correct:
            hybrid_correct += 1

        print(
            f"Complex Q: '{q.question[:45]}...' -> Vector: {vec_res.is_correct} | Graph: {graph_res.is_correct} | Hybrid: {hybrid_res.is_correct}"
        )

    print(
        f"Complex Category Totals (out of {len(complex_questions)}):\n"
        f"  Vector-only: {vector_correct}\n"
        f"  Graph-only:  {graph_correct}\n"
        f"  Hybrid:      {hybrid_correct}"
    )

    assert (
        hybrid_correct > vector_correct
    ), f"Hybrid necessity proof failed: Hybrid ({hybrid_correct}) must be > Vector ({vector_correct})"
    assert (
        hybrid_correct > graph_correct
    ), f"Hybrid necessity proof failed: Hybrid ({hybrid_correct}) must be > Graph ({graph_correct})"
    print("OK: Hybrid Necessity Proof Passed!")
    return hybrid_correct, vector_correct, graph_correct


def test_4_deduplication(retriever: GraphRAGHybridRetriever) -> None:
    print("\n--- Test 4: Deduplication Proof ---")
    # Identify a question where vector search and graph search both surface Paragraph 0
    target_para = "Dana Voss founded Trailmark Robotics in 2018 after years of research in autonomous navigation software."
    question = "Who founded Trailmark Robotics in 2018?"

    _, context_used = retriever.retrieve_context(question, method="hybrid")
    occurrences = context_used.count(target_para)

    print(f"Target paragraph occurrences in merged context: {occurrences}")
    assert occurrences == 1, f"Deduplication proof failed: expected exactly 1 occurrence, found {occurrences}"
    print("OK: Deduplication Proof Passed!")


def main() -> None:
    print("================================================================================")
    print("      LAB 6.3 GRAPHRAG HYBRID RETRIEVAL - SMOKE TEST SUITE      ")
    print("================================================================================")

    router = QueryRouter(use_real=False)
    retriever = GraphRAGHybridRetriever(use_real=False)

    accuracy_count = test_1_router_accuracy(router)
    test_2_category_advantage(retriever)
    hybrid_count, vec_count, graph_count = test_3_hybrid_necessity(retriever)
    test_4_deduplication(retriever)

    print("\n================================================================================")
    print("ALL 4 SMOKE TEST SUITE PROOFS PASSED SUCCESSFULLY!")
    print(f"Router Accuracy: {accuracy_count}/15")
    print(f"Complex Category: Hybrid ({hybrid_count}/5) vs Vector ({vec_count}/5) vs Graph ({graph_count}/5)")
    print("================================================================================")


if __name__ == "__main__":
    main()
