"""
corpus.py — Extended corpus for Lab 6.3 GraphRAG Hybrid Retrieval.

Combines the 20 relational/filler paragraphs from Lab 6.2 with 8 new descriptive paragraphs.
The 8 additional paragraphs contain factual details (e.g., prior professions, company specialties)
that intentionally produce NO graph edges (no works_at, founded_by, located_in, or part_of predicates).
This structural gap allows us to benchmark vector retrieval vs. graph retrieval vs. hybrid retrieval.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

LAB62_DIR = Path(__file__).resolve().parent.parent / "lab-6-2-knowledge-graph-agent"

# Load Lab 6.2 sample_corpus using importlib to avoid sys.path collision
spec = importlib.util.spec_from_file_location("lab62_sample_corpus", LAB62_DIR / "sample_corpus.py")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load sample_corpus.py from {LAB62_DIR}")
sample_corpus_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample_corpus_mod)
LAB62_PARAGRAPHS: list[str] = sample_corpus_mod.CORPUS_PARAGRAPHS

# 8 Additional paragraphs: descriptive text with no relational edges matching Lab 6.2 predicates.
# These create a deliberate structural gap where graph traversal alone cannot find answers.
ADDITIONAL_PARAGRAPHS: list[str] = [
    "Dana Voss worked as a mechanical engineer for eight years before founding Trailmark Robotics.",
    "Trailmark Robotics specializes in building warehouse automation robots.",
    "Rina Achebe was a satellite data scientist before she founded Glacier Analytics.",
    "Glacier Analytics focuses on analyzing agricultural satellite imagery.",
    "Owen Kessler previously worked as a data engineer before joining Glacier Analytics.",
    "Priya Nandan trained as a logistics coordinator before joining Cobalt Freight.",
    "Cobalt Freight specializes in cross-border shipping logistics.",
    "Vantage Industries was established as a holding company for robotics and logistics ventures.",
]

# Combined 28-paragraph corpus
FULL_CORPUS_PARAGRAPHS: list[str] = LAB62_PARAGRAPHS + ADDITIONAL_PARAGRAPHS
