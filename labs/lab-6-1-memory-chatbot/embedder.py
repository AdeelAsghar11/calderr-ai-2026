"""
embedder.py — Local semantic embedder using sentence-transformers.

Reuses the repo's established pattern from lab-3-1-semantic-search:
- Model: all-MiniLM-L6-v2 (384 dimensions)
- normalize_embeddings=True so L2-normalized cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryEmbedder:
    """Thin singleton-friendly wrapper around SentenceTransformer for embedding memory texts."""

    _instance: MemoryEmbedder | None = None
    MODEL_NAME: str = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME)

    @classmethod
    def get_instance(cls) -> MemoryEmbedder:
        """Reuse loaded model instance across memory operations for performance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embedding vectors.

        Returns array of shape (len(texts), 384) with float32 values.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized vectors.
        Because vectors are L2-normalized, dot product equals cosine similarity.
        """
        return float(np.dot(a, b))
