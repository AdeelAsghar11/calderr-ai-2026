"""
hybrid.py — Hybrid BM25 + Semantic Retrieval for the Personal Knowledge Base

Fixes the retrieval misses found in the first run of generate_qa_examples.py:

  Q6 (RAGAS metrics) — the chunk containing the exact phrase "RAGAS metrics:
  faithfulness, answer relevancy, context precision, context recall" existed
  in CalderR_Week-3.pdf but didn't make the top-5 semantic results. A nearby
  chunk (the assessment QUESTION asking to name three RAGAS metrics) scored
  closer on embedding similarity than the chunk containing the actual answer.
  BM25 catches this immediately — "RAGAS metrics" is a direct keyword match.

  Q9 / Q12 — similar pattern: the right document was retrieved, but not
  always the right chunk within it, because pure semantic search ranks by
  embedding distance alone with no signal from literal term overlap.

Same pattern as lab-3-3/hybrid_retriever.py, pointed at this project's own
ChromaDB collection (personal_kb) instead of the CalderR wiki_docs collection.

Fusion: Reciprocal Rank Fusion (RRF) via LangChain's EnsembleRetriever —
a chunk ranked high by EITHER BM25 or semantic search scores well; a chunk
ranked high by BOTH scores highest.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

# pyrefly: ignore [missing-import]
from ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

console = Console()


class PersonalHybridRetriever:
    """
    BM25 (keyword) + semantic (vector) hybrid search over the personal
    knowledge base, fused with Reciprocal Rank Fusion.

    Loads the full corpus into memory once at construction (needed to build
    the BM25 index) — reuse one instance across multiple queries rather than
    constructing it fresh per question (see chat() and generate_qa_examples.py).
    """

    def __init__(
        self,
        bm25_weight:   float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        console.print("[dim]Loading hybrid retriever (BM25 + semantic)...[/dim]")

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        col    = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

        count = col.count()
        data  = col.get(limit=count, include=["documents", "metadatas"])
        texts     = data["documents"]
        metadatas = data["metadatas"]

        # BM25 — keyword matching, catches exact-phrase content semantic search misses
        self._bm25    = BM25Retriever.from_texts(texts, metadatas=metadatas)
        self._bm25.k  = 10

        # Semantic — vector similarity via LangChain's Chroma wrapper, same collection
        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        chroma_lc     = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=hf_embeddings,
        )
        self._vector = chroma_lc.as_retriever(search_kwargs={"k": 10})

        # RRF fusion — a chunk ranked high by BOTH retrievers wins
        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25, self._vector],
            weights=[bm25_weight, vector_weight],
        )

    def retrieve(
        self,
        query:  str,
        top_k:  int = 8,
        source: str | None = None,
    ) -> list[Document]:
        """
        Hybrid retrieval with optional post-hoc source filtering.

        Source filtering happens AFTER fusion rather than before, since
        BM25Retriever has no native metadata filter. This costs nothing in
        practice — the fused candidate pool is small (~15-20) before slicing.
        """
        docs = self._ensemble.invoke(query)

        if source:
            docs = [d for d in docs if d.metadata.get("source") == source]

        return docs[:top_k]


# ── Quick test — verify the exact query that failed before now succeeds ────────
if __name__ == "__main__":
    retriever = PersonalHybridRetriever()
    query     = "what is the RAGAS framework and which metrics does it include?"
    docs      = retriever.retrieve(query, top_k=8)

    print(f"\nQuery: {query}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.metadata.get('filename', '?')}] {d.page_content[:150]}...")