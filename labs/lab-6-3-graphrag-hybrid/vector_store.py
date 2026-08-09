"""
vector_store.py — ChromaDB vector retrieval using SentenceTransformers.

Indexes all 28 paragraphs from FULL_CORPUS_PARAGRAPHS using all-MiniLM-L6-v2 embeddings.
Retrieves top-k closest paragraphs by cosine similarity.
"""

from __future__ import annotations

from typing import List
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from .corpus import FULL_CORPUS_PARAGRAPHS
except ImportError:
    from corpus import FULL_CORPUS_PARAGRAPHS


class VectorRetriever:
    """ChromaDB-backed vector search over extended corpus paragraphs."""

    def __init__(self, corpus: List[str] = FULL_CORPUS_PARAGRAPHS) -> None:
        self.corpus = corpus
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # In-memory ChromaDB client for fast, self-contained indexing
        self.chroma_client = chromadb.Client()
        self.collection_name = "lab_6_3_corpus"

        # Reset collection if exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Compute embeddings and index all paragraphs
        embeddings = self.embedder.encode(
            self.corpus,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [f"doc_{i}" for i in range(len(self.corpus))]
        metadatas = [{"paragraph_id": i} for i in range(len(self.corpus))]

        self.collection.add(
            documents=self.corpus,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k most relevant paragraphs for a natural language query using cosine similarity.
        """
        query_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=min(top_k, len(self.corpus)),
        )

        documents = results.get("documents", [[]])[0]
        return list(documents)
