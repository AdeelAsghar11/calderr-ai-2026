"""
embedder.py — Semantic Embedder wrapper around sentence-transformers.

Key design choices:
- normalize_embeddings=True so cosine similarity reduces to a dot product (faster)
- Embeddings cached to .cache/ as .npy files to avoid re-computing on every run
- Both models produce 384-dimensional vectors; bge uses a query prefix for asymmetric search
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Available models ────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "minilm": "all-MiniLM-L6-v2",
    "bge":    "BAAI/bge-small-en-v1.5",
}

# BGE requires a query prefix for asymmetric retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SemanticEmbedder:
    """Thin wrapper that handles model loading, batched encoding, and search."""

    def __init__(self, model_key: str = "minilm") -> None:
        if model_key not in MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(MODELS)}")

        self.model_key  = model_key
        self.model_name = MODELS[model_key]
        self._model: SentenceTransformer | None = None  # lazy-loaded

    # ── Lazy model loading ───────────────────────────────────────────────────────
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ── Encoding ─────────────────────────────────────────────────────────────────
    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into L2-normalised embedding vectors.

        When is_query=True and the model is BGE, the required query prefix is
        prepended automatically.  Corpus sentences are encoded without a prefix.
        Normalisation means cosine similarity == dot product (faster at scale).
        """
        if is_query and self.model_key == "bge":
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    # ── Similarity ───────────────────────────────────────────────────────────────
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Because ||a|| = ||b|| = 1 after normalisation, cos(θ) == a · b.
        Range: -1 (opposite) → 0 (orthogonal) → 1 (identical direction).
        """
        return float(np.dot(a, b))

    # ── Search ───────────────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        sentences: List[str],
        top_k: int = 5,
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Embed a query and return the top-k most similar corpus sentences.

        Uses vectorised matrix–vector multiply (O(n·d)) instead of a Python loop,
        so it stays fast even with large corpora.

        Returns:
            results  — list of (sentence, similarity_score) tuples
            latency  — wall-clock seconds for the full search
        """
        t0 = time.perf_counter()

        query_emb = self.embed([query], is_query=True)[0]  # shape: (d,)
        scores    = corpus_embeddings @ query_emb          # shape: (n,)  dot product per sentence
        top_idx   = np.argsort(scores)[::-1][:top_k]

        results = [(sentences[i], float(scores[i])) for i in top_idx]
        latency = time.perf_counter() - t0
        return results, latency

    def __repr__(self) -> str:
        return f"SemanticEmbedder(model={self.model_name!r})"
