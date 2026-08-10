"""
router.py — Pre-retrieval Query Router using ChatGroq (llama-3.3-70b-versatile).

Classifies natural language questions into factual, relational, or complex categories.
"""

from __future__ import annotations

import os
from typing import Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    # pyrefly: ignore [missing-import]  
    from .models import QuestionRecord, RouterDecision
except ImportError:
    # pyrefly: ignore [missing-import]
    from models import QuestionRecord
    # pyrefly: ignore [missing-import]
    from models import RouterDecision

# Load environment variables from .env in repository root
load_dotenv()


class QueryRouter:
    """Pre-retrieval Query Router powered by ChatGroq LLM."""

    def __init__(self, use_real: bool = True) -> None:
        self.use_real = use_real
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required. Ensure .env is in the root directory.")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def classify(self, question: str) -> Literal["factual", "relational", "complex"]:
        """Classify a natural language question into 'factual', 'relational', or 'complex' using ChatGroq."""
        prompt = (
            "You are an expert database query router.\n"
            "Classify the user question into exactly one of three categories:\n"
            "- 'factual': questions asking for static descriptive properties or background facts of a single entity (e.g. profession, specialty).\n"
            "- 'relational': questions requiring multi-hop graph path connections across entities (e.g. parent company, founder, city location).\n"
            "- 'complex': questions requiring both a multi-hop relational path AND a descriptive factual detail about an entity reached along the path.\n\n"
            f"Question: {question}\n\n"
            "Return ONLY one word: factual, relational, or complex."
        )

        res = self.llm.invoke(prompt)
        cat_str = str(res.content).strip().lower()

        if "complex" in cat_str:
            return "complex"
        elif "relational" in cat_str:
            return "relational"
        elif "factual" in cat_str:
            return "factual"

        return "complex"

    def route(self, question_record: QuestionRecord) -> RouterDecision:
        """Evaluate question against router classification rules and produce RouterDecision."""
        predicted = self.classify(question_record.question)
        correct = (predicted == question_record.category)

        return RouterDecision(
            question=question_record.question,
            predicted_category=predicted,
            actual_category=question_record.category,
            correct=correct,
        )
