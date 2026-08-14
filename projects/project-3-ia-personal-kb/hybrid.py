"""
hybrid.py — Hybrid BM25 + Semantic Retrieval + Cross-Encoder Re-ranking

Three-stage retrieval, each stage fixing a distinct failure mode found
while building this project:

  Q6 (RAGAS metrics) — the chunk containing the exact phrase "RAGAS metrics:
  faithfulness, answer relevancy, context precision, context recall" was
  CONFIRMED intact and unscrambled in the database (see debug_chunks.py).
  It still didn't surface because the query says "framework" and that exact
  chunk never uses that word — while two OTHER chunks (the Recommended
  Resources table entries) pair "RAGAS" with "framework" directly. BM25
  rewards literal term overlap, so it ranked those chunks higher even
  though they don't actually answer the question. This needed a
  cross-encoder: a model that reads the query and a candidate chunk
  TOGETHER and judges relevance by meaning, not shared words.

  Q9 / Q12 (fixed by hybrid alone) — right document, wrong chunk within it,
  or an exact phrase pure semantic search under-ranked. BM25 fixed these.

Pipeline: BM25 (k=15) + semantic (k=15) → RRF fusion → cross-encoder
re-rank → final top_k. The wider k=15 (vs. the original k=10) exists
specifically to give the re-ranker enough candidates to work with —
a chunk that's semantically correct but ranks 12th on keyword overlap
alone needs to be IN the pool before the cross-encoder can promote it.

Same pattern as lab-3-3/hybrid_retriever.py + reranker.py combined,
pointed at this project's own ChromaDB collection (personal_kb).
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

# EnsembleRetriever moved from langchain.retrievers to langchain_classic.retrievers
# in LangChain v1.0+. Try the new location first, fall back to the old one so
# this works on either version without needing to know which is installed.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    # pyrefly: ignore [missing-import]
    from langchain.retrievers import EnsembleRetriever

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
        use_reranker:  bool  = True,
    ) -> None:
        console.print("[dim]Loading hybrid retriever (BM25 + semantic)...[/dim]")

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        col    = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

        count = col.count()
        data  = col.get(limit=count, include=["documents", "metadatas"])
        texts     = data["documents"]
        metadatas = data["metadatas"]

        # BM25 — keyword matching, catches exact-phrase content semantic search misses.
        # k=15 (not 10) so the candidate pool feeding the re-ranker is wide enough
        # to include chunks that rank poorly on literal keyword overlap but are
        # still the semantically correct answer.
        self._bm25    = BM25Retriever.from_texts(texts, metadatas=metadatas)
        self._bm25.k  = 15

        # Semantic — vector similarity via LangChain's Chroma wrapper, same collection
        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        chroma_lc     = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=hf_embeddings,
        )
        self._vector = chroma_lc.as_retriever(search_kwargs={"k": 15})

        # RRF fusion — a chunk ranked high by BOTH retrievers wins
        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25, self._vector],
            weights=[bm25_weight, vector_weight],
        )

        # Cross-encoder re-ranker — reads query + chunk TOGETHER and scores true
        # relevance, catching cases where the right chunk uses different wording
        # than the query (e.g. query says "framework", chunk says "metrics:").
        # BM25 and vector search both score chunks independently of each other;
        # only a cross-encoder does joint attention over the pair.
        self._reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            console.print("[dim]Loading cross-encoder re-ranker (BAAI/bge-reranker-base)...[/dim]")
            self._reranker = CrossEncoder("BAAI/bge-reranker-base")

    def retrieve(
        self,
        query:  str,
        top_k:  int = 8,
        source: str | None = None,
    ) -> list[Document]:
        """
        Hybrid retrieval (BM25 + semantic, RRF fusion), optionally re-ranked
        by a cross-encoder before final top_k slicing.

        Source filtering happens AFTER fusion rather than before, since
        BM25Retriever has no native metadata filter. This costs nothing in
        practice — the fused candidate pool is small (~20-30) before slicing.
        """
        docs = self._ensemble.invoke(query)

        if source:
            docs = [d for d in docs if d.metadata.get("source") == source]

        if self._reranker and docs:
            pairs  = [(query, d.page_content) for d in docs]
            scores = self._reranker.predict(pairs)
            docs   = [d for _, d in sorted(zip(scores.tolist(), docs), key=lambda x: x[0], reverse=True)]

        return docs[:top_k]


# ── Quick test — verify the exact query that failed before now succeeds ────────
if __name__ == "__main__":
    retriever = PersonalHybridRetriever()
    query     = "what is the RAGAS framework and which metrics does it include?"
    docs      = retriever.retrieve(query, top_k=8)

    print(f"\nQuery: {query}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.metadata.get('filename', '?')}] {d.page_content[:150]}...")

    found = any("faithfulness" in d.page_content.lower() for d in docs)
    print(f"\n{'✓ FIXED' if found else '✗ still missing'} — chunk with RAGAS metrics list "
          f"{'is now' if found else 'is NOT'} in top-8")