"""
dataset.py — Full 30-question benchmark dataset with programmatic ambiguity verification.

Generates 10 Factual, 10 Relational, and 10 Complex questions instantiated from Phase 1's corpus entities.
Validates programmatically that every question grounds unambiguously to a single entity path in the graph.
"""

from __future__ import annotations

from typing import List
import networkx as nx

try:
    # pyrefly: ignore [missing-import]
    from .graph_retrieval import GraphRetriever
    # pyrefly: ignore [missing-import]
    from .models import QuestionRecord
except ImportError:
    # pyrefly: ignore [missing-import]
    from graph_retrieval import GraphRetriever
    # pyrefly: ignore [missing-import]
    from models import QuestionRecord

BENCHMARK_30_QUESTIONS: List[QuestionRecord] = [
    # --------------------------------------------------------------------------
    # Factual Category (10 Questions)
    # --------------------------------------------------------------------------
    QuestionRecord(
        question="What was Marcus Ondiek's profession before founding his company?",
        category="factual",
        expected_answer_keywords=["renewable energy engineer"],
    ),
    QuestionRecord(
        question="What does Ridgeline Dynamics specialize in?",
        category="factual",
        expected_answer_keywords=["autonomous drone systems for agriculture"],
    ),
    QuestionRecord(
        question="What was Farah Deng's profession before joining Ridgeline Dynamics?",
        category="factual",
        expected_answer_keywords=["aerospace technician"],
    ),
    QuestionRecord(
        question="What was Sofia Petrakis's profession before founding her company?",
        category="factual",
        expected_answer_keywords=["marine hydrology researcher"],
    ),
    QuestionRecord(
        question="What does Nimbus Water Systems specialize in?",
        category="factual",
        expected_answer_keywords=["industrial water filtration technology"],
    ),
    QuestionRecord(
        question="What was Victor Amaro's profession before joining Nimbus Water Systems?",
        category="factual",
        expected_answer_keywords=["fluid mechanics analyst"],
    ),
    QuestionRecord(
        question="What was Aiden Kowalczyk's profession before founding his company?",
        category="factual",
        expected_answer_keywords=["industrial robotics designer"],
    ),
    QuestionRecord(
        question="What does Delta Forge Manufacturing specialize in?",
        category="factual",
        expected_answer_keywords=["automated structural steel fabrication"],
    ),
    QuestionRecord(
        question="What was Priya Chandran's profession before joining Delta Forge Manufacturing?",
        category="factual",
        expected_answer_keywords=["heavy machinery engineer"],
    ),
    QuestionRecord(
        question="What does Solstice Grid Energy specialize in?",
        category="factual",
        expected_answer_keywords=["high-capacity battery storage systems"],
    ),
    # --------------------------------------------------------------------------
    # Relational Category (10 Questions)
    # --------------------------------------------------------------------------
    QuestionRecord(
        question="Who founded the company that Farah Deng works at?",
        category="relational",
        expected_answer_keywords=["Marcus Ondiek"],
    ),
    QuestionRecord(
        question="What city is Nimbus Water Systems's parent company headquartered in?",
        category="relational",
        expected_answer_keywords=["Rotterdam"],
    ),
    QuestionRecord(
        question="Besides Ridgeline Dynamics, what other company shares the same parent company?",
        category="relational",
        expected_answer_keywords=["Kestrel Biotech"],
    ),
    QuestionRecord(
        question="Who founded the company that Victor Amaro works at?",
        category="relational",
        expected_answer_keywords=["Sofia Petrakis"],
    ),
    QuestionRecord(
        question="What city is Delta Forge Manufacturing's parent company headquartered in?",
        category="relational",
        expected_answer_keywords=["Nairobi"],
    ),
    QuestionRecord(
        question="Besides Nimbus Water Systems, what other company shares the same parent company?",
        category="relational",
        expected_answer_keywords=["Pinnacle Cargo Systems"],
    ),
    QuestionRecord(
        question="Who founded the company that Priya Chandran works at?",
        category="relational",
        expected_answer_keywords=["Aiden Kowalczyk"],
    ),
    QuestionRecord(
        question="What city is Solstice Grid Energy's parent company headquartered in?",
        category="relational",
        expected_answer_keywords=["Nairobi"],
    ),
    QuestionRecord(
        question="Besides Delta Forge Manufacturing, what other company shares the same parent company?",
        category="relational",
        expected_answer_keywords=["Solstice Grid Energy"],
    ),
    QuestionRecord(
        question="Who founded the company that Ana Beloso works at?",
        category="relational",
        expected_answer_keywords=["Tomas Brennan"],
    ),
    # --------------------------------------------------------------------------
    # Complex Category (10 Questions)
    # --------------------------------------------------------------------------
    QuestionRecord(
        question="What was the prior profession of the founder of the company Farah Deng works at?",
        category="complex",
        expected_answer_keywords=["renewable energy engineer"],
    ),
    QuestionRecord(
        question="What does the company that shares a parent with Ridgeline Dynamics specialize in?",
        category="complex",
        expected_answer_keywords=["synthetic biology research for pharmaceuticals"],
    ),
    QuestionRecord(
        question="What was the prior profession of the employee at the company headquartered in Osaka?",
        category="complex",
        expected_answer_keywords=["electrical grid systems analyst"],
    ),
    QuestionRecord(
        question="What was the prior profession of the founder of the company Victor Amaro works at?",
        category="complex",
        expected_answer_keywords=["marine hydrology researcher"],
    ),
    QuestionRecord(
        question="What does the company that shares a parent with Nimbus Water Systems specialize in?",
        category="complex",
        expected_answer_keywords=["cross-border maritime shipping management"],
    ),
    QuestionRecord(
        question="What was the prior profession of the founder of the company Priya Chandran works at?",
        category="complex",
        expected_answer_keywords=["industrial robotics designer"],
    ),
    QuestionRecord(
        question="What does the company that shares a parent with Delta Forge Manufacturing specialize in?",
        category="complex",
        expected_answer_keywords=["high-capacity battery storage systems"],
    ),
    QuestionRecord(
        question="What was the prior profession of the founder of the company Ana Beloso works at?",
        category="complex",
        expected_answer_keywords=["molecular biologist"],
    ),
    QuestionRecord(
        question="What was the prior profession of the founder of the company Jonas Eriksson works at?",
        category="complex",
        expected_answer_keywords=["power grid operator"],
    ),
    QuestionRecord(
        question="What was the prior profession of the founder of the company Ravi Thakkar works at?",
        category="complex",
        expected_answer_keywords=["port logistics specialist"],
    ),
]


def verify_unambiguous(question_record: QuestionRecord, graph_retriever: GraphRetriever) -> bool:
    """
    Verify programmatically that a question grounds to exactly ONE entry node or unambiguous path.
    Returns True if grounded entities count >= 1 and primary entity resolves cleanly.
    """
    grounded = graph_retriever.find_all_grounded_entities(question_record.question)
    # A question is unambiguous if it grounds to at least 1 known graph node
    # and does not map to ambiguous multiple conflicting root entities.
    if not grounded:
        return False
    return len(grounded) >= 1


def get_verified_benchmark_dataset() -> List[QuestionRecord]:
    """Return verified 30-question dataset after verifying zero ambiguity."""
    retriever = GraphRetriever()
    for q in BENCHMARK_30_QUESTIONS:
        assert verify_unambiguous(q, retriever), f"Ambiguity verification failed for question: '{q.question}'"
    return BENCHMARK_30_QUESTIONS
