"""
semantic_store.py — ChromaDB-backed semantic store with recency + relevance hybrid scoring.

Indexes episodic turns and profile facts as 384-dimensional all-MiniLM-L6-v2 embeddings.
Implements composite memory scoring: score = min_max(recency) + min_max(relevance).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chromadb
import numpy as np

try:
    from .embedder import MemoryEmbedder
    from .models import InteractionLog, RetrievedMemory
except ImportError:
    from embedder import MemoryEmbedder
    from models import InteractionLog, RetrievedMemory

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class SemanticStore:
    """ChromaDB persistent semantic vector store for hybrid recency+relevance retrieval."""

    def __init__(self, chroma_path: str | Path | None = None) -> None:
        if chroma_path is None:
            chroma_path = DEFAULT_DATA_DIR / "chroma_db"
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="research_memories",
            metadata={"description": "Semantic embeddings of research turns and facts"},
        )
        self.embedder = MemoryEmbedder.get_instance()

    def add_log_embedding(self, log: InteractionLog) -> None:
        """Embed and index an InteractionLog in ChromaDB using its unique database ID."""
        if log.id is None:
            return

        emb = self.embedder.embed([log.content])[0].tolist()
        self.collection.upsert(
            ids=[str(log.id)],
            embeddings=[emb],
            documents=[log.content],
            metadatas=[{
                "session_id": log.session_id,
                "timestamp": log.timestamp,
                "role": log.role,
                "importance_score": log.importance_score,
            }],
        )

    def _min_max_scale(self, values: List[float]) -> List[float]:
        """Scale a list of floats to [0.0, 1.0] range using min-max scaling."""
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if abs(max_v - min_v) < 1e-9:
            return [1.0 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]

    def retrieve_memories(
        self,
        query: str,
        current_session_id: str,
        candidate_logs: List[InteractionLog],
        top_k: int = 5,
        now: Optional[datetime] = None,
    ) -> List[RetrievedMemory]:
        """
        Retrieve and rank top_k past memories matching query using recency + relevance scoring.
        Excludes entries from current_session_id.
        """
        # Filter candidate logs to exclude current session
        cands = [log for log in candidate_logs if log.session_id != current_session_id]
        if not cands:
            return []

        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        query_emb = self.embedder.embed([query])[0]

        cand_ids = [str(c.id) for c in cands if c.id is not None]
        chroma_res = self.collection.get(ids=cand_ids, include=["embeddings"])
        embeddings_by_id = {}
        if chroma_res and chroma_res.get("ids"):
            for cid, emb in zip(chroma_res["ids"], chroma_res["embeddings"]):
                embeddings_by_id[int(cid)] = np.array(emb, dtype=np.float32)

        raw_recencies: List[float] = []
        raw_relevances: List[float] = []

        for cand in cands:
            # Parse timestamp and compute hours since turn creation
            try:
                cand_dt = datetime.fromisoformat(cand.timestamp)
            except Exception:
                cand_dt = now
            if cand_dt.tzinfo is None:
                cand_dt = cand_dt.replace(tzinfo=timezone.utc)

            hours_since = max(0.0, (now - cand_dt).total_seconds() / 3600.0)
            raw_rec = 0.995 ** hours_since
            raw_recencies.append(raw_rec)

            if cand.id in embeddings_by_id:
                c_emb = embeddings_by_id[cand.id]
            else:
                c_emb = self.embedder.embed([cand.content])[0]

            raw_rel = self.embedder.cosine_similarity(query_emb, c_emb)
            raw_relevances.append(raw_rel)

        norm_recencies = self._min_max_scale(raw_recencies)
        norm_relevances = self._min_max_scale(raw_relevances)

        results: List[RetrievedMemory] = []
        for i, cand in enumerate(cands):
            norm_rec = norm_recencies[i]
            norm_rel = norm_relevances[i]
            comp_score = norm_rec + norm_rel

            results.append(
                RetrievedMemory(
                    log=cand,
                    recency_score=norm_rec,
                    relevance_score=norm_rel,
                    composite_score=comp_score,
                )
            )

        # Sort descending by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results[:top_k]
