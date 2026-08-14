"""
smoke_test.py — Automated verification suite for Lab 6.2 Knowledge Graph Agent.

Executes all 5 required proof cases:
1. Deduplication proof (verifies entities across 20 paragraphs merge into single canonical nodes).
2. Multi-hop correctness proof (verifies all 5 validation questions with path lengths >= 2).
3. Undirected traversal proof (verifies Q1 traversing reversed edge orientation).
4. Keyword search fails proof (verifies plain keyword search fails on at least 2 of 5 questions).
5. Visualization proof (verifies Pyvis HTML generation and entity presence in HTML file).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    # pyrefly: ignore [missing-import]
    from .graph_builder import build_knowledge_graph, render_pyvis_graph
    # pyrefly: ignore [missing-import]
    from .query_agent import KnowledgeGraphQueryAgent
    # pyrefly: ignore [missing-import]
    from .sample_corpus import CORPUS_PARAGRAPHS
except ImportError:
    # pyrefly: ignore [missing-import]
    from graph_builder import build_knowledge_graph, render_pyvis_graph
    # pyrefly: ignore [missing-import]
    from query_agent import KnowledgeGraphQueryAgent
    # pyrefly: ignore [missing-import]
    from sample_corpus import CORPUS_PARAGRAPHS


VALIDATION_QUESTIONS = [
    ("What city is Dana Voss's company located in?", "Austin"),
    ("Who founded the company that Owen Kessler works at?", "Rina Achebe"),
    ("What city is Cobalt Freight's parent company headquartered in?", "Seattle"),
    ("Besides Trailmark Robotics, what other company shares the same parent company?", "Cobalt Freight"),
    ("What city does Dana Voss's company's parent company operate in?", "Seattle"),
]


def run_smoke_tests() -> None:
    print("=" * 70)
    print("RUNNING LAB 6.2 KNOWLEDGE GRAPH AGENT SMOKE TESTS")
    print("=" * 70)

    # Build knowledge graph in stub mode
    print("\nBuilding Knowledge Graph from sample corpus...")
    graph = build_knowledge_graph(CORPUS_PARAGRAPHS, use_real=False)
    agent = KnowledgeGraphQueryAgent(graph=graph, corpus=CORPUS_PARAGRAPHS, use_real=False)

    # -------------------------------------------------------------------------
    # CASE 1: The Deduplication Proof
    # -------------------------------------------------------------------------
    print("\n[CASE 1] Running Entity Deduplication Proof...")
    num_nodes = graph.number_of_nodes()
    print(f"Total graph nodes after merging: {num_nodes}")

    assert "Trailmark Robotics" in graph.nodes, "Case 1 Failure: 'Trailmark Robotics' node missing!"
    trailmark_pids = graph.nodes["Trailmark Robotics"]["source_paragraph_ids"]
    print(f"  'Trailmark Robotics' source paragraph IDs: {trailmark_pids}")

    assert len(trailmark_pids) > 1, (
        f"Case 1 Failure: Expected 'Trailmark Robotics' to be merged across multiple paragraphs, got {trailmark_pids}"
    )
    assert num_nodes == 12, f"Case 1 Failure: Expected 12 deduplicated entity nodes, got {num_nodes}"
    print("[OK] CASE 1 PASSED: Entities merged cleanly into deduplicated canonical nodes.")

    # -------------------------------------------------------------------------
    # CASE 2 & 4: Multi-Hop Correctness & Keyword Search Fails Proofs
    # -------------------------------------------------------------------------
    print("\n[CASE 2 & 4] Running Multi-Hop Correctness & Keyword Search Baseline Proofs...")

    correct_count = 0
    keyword_fail_count = 0

    for idx, (question, expected_ans) in enumerate(VALIDATION_QUESTIONS, start=1):
        result = agent.answer_query(question)

        path_str = " -> ".join([f"{h.from_entity} --({h.relationship} [{h.direction}])--> {h.to_entity}" for h in result.path])
        print(f"\nQuestion {idx}: {question!r}")
        print(f"  Result Answer: {result.answer!r} (Expected: {expected_ans!r})")
        print(f"  Reasoning Path ({len(result.path)} hops): {path_str}")
        print(f"  Keyword Search Would Succeed: {result.keyword_search_would_succeed}")

        # Check answer match
        if expected_ans.lower() in result.answer.lower():
            correct_count += 1

        # Check path length
        assert len(result.path) >= 2, f"Question {idx} Failure: Path length must be >= 2 hops, got {len(result.path)}"

        # Check keyword failure
        if not result.keyword_search_would_succeed:
            keyword_fail_count += 1

    assert correct_count >= 4, f"Case 2 Failure: Expected at least 4/5 questions answered correctly, got {correct_count}/5"
    print(f"[OK] CASE 2 PASSED: {correct_count}/5 questions answered correctly with path length >= 2.")

    assert keyword_fail_count >= 2, (
        f"Case 4 Failure: Expected keyword search to fail on at least 2/5 questions, but failed on {keyword_fail_count}/5"
    )
    print(f"[OK] CASE 4 PASSED: Keyword search baseline failed on {keyword_fail_count}/5 questions (proving graph value).")

    # -------------------------------------------------------------------------
    # CASE 3: Undirected Traversal Proof (Question 1)
    # -------------------------------------------------------------------------
    print("\n[CASE 3] Running Undirected Traversal Proof (Question 1)...")
    q1_result = agent.answer_query(VALIDATION_QUESTIONS[0][0])
    assert len(q1_result.path) >= 2, "Q1 path length must be >= 2"
    first_hop = q1_result.path[0]
    print(f"  Q1 First Hop: {first_hop.from_entity} --({first_hop.relationship} [{first_hop.direction}])--> {first_hop.to_entity}")

    assert first_hop.direction == "reverse", (
        f"Case 3 Failure: First hop direction for Q1 should be 'reverse', got {first_hop.direction!r}"
    )
    print("[OK] CASE 3 PASSED: Undirected search traversed reversed edge orientation correctly.")

    # -------------------------------------------------------------------------
    # CASE 5: Visualization Proof
    # -------------------------------------------------------------------------
    print("\n[CASE 5] Running Pyvis HTML Visualization Proof...")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_html = Path(temp_dir) / "test_graph.html"
        out_file = render_pyvis_graph(graph, output_file=temp_html)

        assert out_file.exists(), "Case 5 Failure: Pyvis HTML file was not created!"
        file_text = out_file.read_text(encoding="utf-8")
        assert "Trailmark Robotics" in file_text, "Case 5 Failure: 'Trailmark Robotics' not found in HTML text!"
        assert "Vantage Industries" in file_text, "Case 5 Failure: 'Vantage Industries' not found in HTML text!"
        print(f"  Pyvis HTML file generated ({out_file.stat().st_size} bytes)")
        print("[OK] CASE 5 PASSED: Interactive Pyvis HTML successfully rendered with entity node text.")

    print("\n" + "=" * 70)
    print("ALL 5 SMOKE TEST PROOF CASES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_tests()
