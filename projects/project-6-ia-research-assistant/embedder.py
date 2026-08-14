"""
embedder.py — Local semantic embedder using sentence-transformers (all-MiniLM-L6-v2).

Self-contained wrapper producing L2-normalized 384-dimensional embeddings for memory retrieval.
"""

from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryEmbedder:
    """Singleton-friendly wrapper around SentenceTransformer all-MiniLM-L6-v2."""

    _instance: MemoryEmbedder | None = None
    MODEL_NAME: str = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME)

    @classmethod
    def get_instance(cls) -> MemoryEmbedder:
        """Reuse loaded model instance across embedding calls for performance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into L2-normalized embedding vectors.
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
        Cosine similarity between two L2-normalized embedding vectors.
        Since vectors are normalized, dot product equals cosine similarity.
        """
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.dot(a, b))
