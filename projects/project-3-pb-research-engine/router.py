"""
router.py — Knowledge Router

This is the brain of the research engine. For every query it answers
one question: where should we look for information?

Three options:
  local  — search our ChromaDB knowledge base (AI/ML topics we ingested)
  web    — search the live internet via DuckDuckGo
  both   — retrieve from both sources and merge

How it decides:
  An LLM reads the query and classifies it. The LLM is given a clear
  description of what topics are covered locally so it can make an
  informed decision. It outputs structured JSON so we can parse it
  reliably without string matching.

Why this matters:
  - Sending every query to the web is slow and returns noisy results
    for topics already covered locally
  - Sending every query to ChromaDB fails for current events
  - The router gives us the best of both worlds automatically

Examples:
  "what is backpropagation?"           → local  (covered in our docs)
  "latest GPT-5 release date?"         → web    (current news)
  "how does RAG compare to GPT-4?"     → both   (concept + current info)
"""

from __future__ import annotations

import json
import os
from enum import Enum

# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

load_dotenv(find_dotenv())

# ── What topics live in our local knowledge base ──────────────────────────────
LOCAL_TOPICS = """
- Artificial intelligence (history, definitions, concepts)
- Machine learning (supervised, unsupervised, algorithms)
- Deep learning (neural networks, backpropagation, architectures)
- Natural language processing (tokenization, NER, sentiment analysis)
- Large language models (GPT, BERT, training methods)
- Transformer architecture (attention mechanism, self-attention, BERT, GPT)
- Vector databases (FAISS, ChromaDB, similarity search, ANN)
- Retrieval augmented generation (RAG, ingestion pipelines, RAGAS)
- Reinforcement learning (reward functions, Q-learning, policy gradient)
- Computer vision (CNNs, image classification, object detection, segmentation)
"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a knowledge router for a research engine.\n\n"
     "The LOCAL knowledge base covers these topics:\n"
     "{local_topics}\n\n"
     "Your job: given a user query, decide the best source:\n"
     "  'local'  — query is fully covered by the local knowledge base\n"
     "  'web'    — query needs current information, news, prices, recent events,\n"
     "             specific products, or people not covered locally\n"
     "  'both'   — query needs foundational knowledge AND current context\n\n"
     "Respond with ONLY valid JSON, no markdown:\n"
     '{{"source": "local|web|both", "reasoning": "one sentence explanation"}}'),
    ("human", "Query: {query}"),
])


# ── Data model for the routing decision ───────────────────────────────────────
class SourceType(str, Enum):
    LOCAL = "local"
    WEB   = "web"
    BOTH  = "both"


class RoutingDecision(BaseModel):
    source:    SourceType
    reasoning: str
    query:     str


# ── Router class ──────────────────────────────────────────────────────────────
class KnowledgeRouter:
    """
    Classifies a query into local / web / both using an LLM.

    The LLM sees the query and a description of local topics.
    It outputs JSON that we parse into a RoutingDecision object.

    If parsing fails (LLM outputs unexpected text), we fall back to 'both'
    so the user always gets an answer even if routing is uncertain.
    """

    def __init__(self) -> None:
        self._llm   = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,           # routing must be deterministic
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self._chain = ROUTER_PROMPT | self._llm | StrOutputParser()

    def route(self, query: str) -> RoutingDecision:
        """
        Route a query. Returns a RoutingDecision with source and reasoning.
        Falls back to 'both' on any parsing error.
        """
        try:
            raw = self._chain.invoke({
                "query":        query,
                "local_topics": LOCAL_TOPICS,
            })
            # Strip markdown fences if the LLM added them anyway
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            return RoutingDecision(
                source    = SourceType(data.get("source", "both")),
                reasoning = data.get("reasoning", ""),
                query     = query,
            )
        except Exception:
            # Safe fallback — never crash on routing failure
            return RoutingDecision(
                source    = SourceType.BOTH,
                reasoning = "Routing parse failed — defaulting to both sources.",
                query     = query,
            )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = KnowledgeRouter()
    tests  = [
        "What is backpropagation?",
        "What is the current price of Nvidia stock?",
        "How does RAG compare to the latest GPT-5 model?",
        "Who won the 2026 World Cup?",
        "Explain the attention mechanism in transformers",
    ]
    for q in tests:
        d = router.route(q)
        print(f"[{d.source.value:5}] {d.reasoning[:70]}")
        print(f"       Q: {q}\n")
