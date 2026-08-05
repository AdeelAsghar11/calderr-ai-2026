"""
Smoke test for document_graph.py.

Uses langchain_core's DeterministicFakeEmbedding instead of the real
HuggingFaceEmbeddings, so this proves the graph wiring, the conditional
edge, and the split/chunk logic all work correctly without needing to
download the actual sentence-transformers model. Swap in the real model
(the default in build_graph()) once you're running this on your machine.

Run: python smoke_test.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_core.embeddings import DeterministicFakeEmbedding

from document_graph import build_graph

FAKE_DIM = 384  # same dimensionality as all-MiniLM-L6-v2, for a realistic check


def run_case(graph, file_path: str, label: str):
    print(f"\n--- {label}: {file_path} ---")
    result = graph.invoke({"file_path": file_path, "parts": []})
    if not result.get("is_valid", False):
        print(f"  REJECTED at validation: {result['validation_message']}")
        return result
    print(f"  is_oversized: {result['is_oversized']}")
    print(f"  parts:        {len(result.get('parts', []))}")
    print(f"  chunks:       {len(result['chunks'])}")
    print(f"  embedded:     {result['num_embedded']} vectors, dim={result['embedding_dim']}")
    print(f"  summary:      {result['summary']}")
    return result


if __name__ == "__main__":
    fake_embedder = DeterministicFakeEmbedding(size=FAKE_DIM)
    graph = build_graph(embedding_model=fake_embedder)

    r1 = run_case(graph, os.path.join(BASE_DIR, "short_note.txt"), "NORMAL branch (under threshold)")
    assert r1["is_valid"] is True
    assert r1["is_oversized"] is False
    assert "parts" not in r1 or r1["parts"] == []
    assert r1["embedding_dim"] == FAKE_DIM

    r2 = run_case(graph, os.path.join(BASE_DIR, "long_report.txt"), "OVERSIZED branch (split first)")
    assert r2["is_valid"] is True
    assert r2["is_oversized"] is True
    assert len(r2["parts"]) > 1
    assert r2["num_embedded"] == len(r2["chunks"])

    r3 = run_case(graph, os.path.join(BASE_DIR, "empty_file.txt"), "INVALID branch (whitespace only)")
    assert r3["is_valid"] is False

    r4 = run_case(graph, os.path.join(BASE_DIR, "does_not_exist.txt"), "INVALID branch (missing file)")
    assert r4["is_valid"] is False

    print("\nAll branches exercised, all assertions passed.")


