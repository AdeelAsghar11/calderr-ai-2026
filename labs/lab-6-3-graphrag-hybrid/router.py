"""
router.py — Pre-retrieval Query Router classifying questions into factual, relational, or complex.

In stub mode: uses explicit, inspectable phrase-cue heuristics.
In real mode: uses ChatGroq (llama-3.3-70b-versatile) zero-shot classification.
"""

from __future__ import annotations

import os
from typing import Literal

try:
    # pyrefly: ignore [missing-import]
    from .models import QuestionRecord, RouterDecision
except ImportError:
    # pyrefly: ignore [missing-import]
    from models import QuestionRecord, RouterDecision


class QueryRouter:
    """Query router for deciding retrieval strategy before execution."""

    def __init__(self, use_real: bool = False) -> None:
        self.use_real = use_real

    def classify(self, question: str) -> Literal["factual", "relational", "complex"]:
        """
        Classify a natural language question into 'factual', 'relational', or 'complex'.
        """
        if self.use_real:
            return self._classify_real(question)
        return self._classify_stub(question)

    def route(self, question_record: QuestionRecord) -> RouterDecision:
        """
        Evaluate question against router classification rules and produce RouterDecision.
        """
        predicted = self.classify(question_record.question)
        correct = (predicted == question_record.category)

        return RouterDecision(
            question=question_record.question,
            predicted_category=predicted,
            actual_category=question_record.category,
            correct=correct,
        )

    def _classify_stub(self, question: str) -> Literal["factual", "relational", "complex"]:
        """
        Inspectable phrase-cue heuristic classification.
        Does NOT look at target answer keywords — operates purely on query syntax/semantics.
        """
        q_lower = question.lower()

        # Factual cues: asking for descriptive details, backgrounds, specialties, or roles
        factual_cues = [
            "profession",
            "specialize",
            "specializes",
            "focus on",
            "focuses on",
            "before founding",
            "profession background",
            "prior profession",
            "what does",
            "what did",
        ]

        # Relational cues: asking for entities connected via graph relationships (founded, located, parent, works_at)
        relational_cues = [
            "who founded",
            "founder of",
            "parent company",
            "shares the same parent",
            "shares a parent",
            "headquartered in",
            "located in",
            "what city",
            "city does",
            "works at",
            "is part of",
            "company that",
        ]

        has_factual = any(cue in q_lower for cue in factual_cues)
        has_relational = any(cue in q_lower for cue in relational_cues)

        if has_factual and has_relational:
            return "complex"
        elif has_factual and not has_relational:
            return "factual"
        elif has_relational and not has_factual:
            return "relational"
        else:
            # Default fallback for unclassified questions
            return "factual"

    def _classify_real(self, question: str) -> Literal["factual", "relational", "complex"]:
        """
        ChatGroq LLM router classification. Raises RuntimeError if GROQ_API_KEY is missing.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required for --real mode.")

        # pyrefly: ignore [missing-import]
        from langchain_groq import ChatGroq

        prompt = (
            "You are a database query router. Classify the user question into exactly one of three categories:\n"
            "- 'factual': questions asking for static descriptive properties or background facts of a single entity (e.g. profession, specialty).\n"
            "- 'relational': questions requiring multi-hop graph path connections across entities (e.g. parent company, founder, city location).\n"
            "- 'complex': questions requiring both a multi-hop relational path AND a descriptive factual detail about an entity reached along the path.\n\n"
            f"Question: {question}\n\n"
            "Return ONLY one word: factual, relational, or complex."
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        res = llm.invoke(prompt)
        cat_str = str(res.content).strip().lower()

        if "complex" in cat_str:
            return "complex"
        elif "relational" in cat_str:
            return "relational"
        elif "factual" in cat_str:
            return "factual"

        return "complex"
