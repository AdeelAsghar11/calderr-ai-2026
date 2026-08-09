"""
hybrid_retriever.py — GraphRAG hybrid retrieval engine for Project 6-PB Phase 1.

Orchestrates vector search, graph traversal, context deduplication, and answer generation.
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Tuple

try:
    from .corpus import FULL_CORPUS_PARAGRAPHS
    from .graph_retrieval import GraphRetriever
    from .models import MethodResult, QuestionRecord
    from .router import QueryRouter
    from .vector_store import VectorRetriever
except ImportError:
    from corpus import FULL_CORPUS_PARAGRAPHS
    from graph_retrieval import GraphRetriever
    from models import MethodResult, QuestionRecord
    from router import QueryRouter
    from vector_store import VectorRetriever


class GraphRAGHybridRetriever:
    """GraphRAG Hybrid Retriever merging ChromaDB vector search and NetworkX graph traversal."""

    def __init__(self, corpus: List[str] = FULL_CORPUS_PARAGRAPHS, use_real: bool = False) -> None:
        self.corpus = corpus
        self.use_real = use_real
        self.vector_retriever = VectorRetriever(corpus=self.corpus)
        self.graph_retriever = GraphRetriever(corpus=self.corpus)
        self.router = QueryRouter(use_real=use_real)

    def merge_and_deduplicate(
        self,
        vector_paras: List[str],
        graph_paras: List[str],
    ) -> List[str]:
        """Merge vector and graph retrieved paragraphs removing exact duplicates while preserving order."""
        seen: set[str] = set()
        merged: List[str] = []

        for p in vector_paras + graph_paras:
            p_strip = p.strip()
            if p_strip not in seen:
                seen.add(p_strip)
                merged.append(p_strip)

        return merged

    def retrieve_context(
        self,
        question: str,
        method: Literal["vector_only", "graph_only", "hybrid"],
    ) -> Tuple[List[str], str]:
        """Retrieve context paragraphs according to specified retrieval method."""
        vector_paras: List[str] = []
        graph_paras: List[str] = []

        if method == "vector_only":
            vector_paras = self.vector_retriever.retrieve(question, top_k=5)
        elif method == "graph_only":
            graph_paras, _ = self.graph_retriever.retrieve(question)
        elif method == "hybrid":
            # 1. Base vector search on full question
            vector_paras = self.vector_retriever.retrieve(question, top_k=5)

            # 2. Graph neighborhood expansion
            graph_paras, related_entities = self.graph_retriever.retrieve(question)

            # 3. Supplemental vector lookups for grounded/traversed entities
            extra_vector_paras: List[str] = []
            for entity in related_entities:
                entity_query = f"{entity} profession specialization background"
                extra_vector_paras.extend(self.vector_retriever.retrieve(entity_query, top_k=2))

            vector_paras.extend(extra_vector_paras)

        dedup_paras = self.merge_and_deduplicate(vector_paras, graph_paras)
        context_str = "\n\n".join(dedup_paras)
        return dedup_paras, context_str

    def generate_answer(self, question: str, context: str) -> str:
        """Generate final answer from retrieved context (stub mode vs ChatGroq real mode)."""
        if not context.strip():
            return "No relevant context found to answer the question."

        if self.use_real:
            return self._generate_answer_real(question, context)

        return f"[Retrieved Context Summary]: {context}"

    def _generate_answer_real(self, question: str, context: str) -> str:
        """Real LLM answer generation using ChatGroq. Raises RuntimeError if GROQ_API_KEY missing."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required for --real mode.")

        from langchain_groq import ChatGroq

        prompt = (
            "You are a helpful QA assistant.\n"
            "Given the following retrieved context, answer the question concisely and accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        res = llm.invoke(prompt)
        return str(res.content).strip()

    def process_question(
        self,
        question_record: QuestionRecord,
        override_method: Optional[Literal["vector_only", "graph_only", "hybrid"]] = None,
    ) -> MethodResult:
        """Execute query pipeline: route question -> retrieve context -> generate answer -> verify correctness."""
        question = question_record.question

        if override_method:
            method = override_method
        else:
            predicted_cat = self.router.classify(question)
            cat_to_method: dict[str, Literal["vector_only", "graph_only", "hybrid"]] = {
                "factual": "vector_only",
                "relational": "graph_only",
                "complex": "hybrid",
            }
            method = cat_to_method.get(predicted_cat, "hybrid")

        _, context_str = self.retrieve_context(question, method=method)
        answer = self.generate_answer(question, context_str)

        is_correct = all(
            kw.lower() in context_str.lower() for kw in question_record.expected_answer_keywords
        )

        return MethodResult(
            method=method,
            context_used=context_str,
            answer=answer,
            is_correct=is_correct,
        )
