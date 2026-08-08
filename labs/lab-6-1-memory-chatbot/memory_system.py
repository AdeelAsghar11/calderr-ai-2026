"""
memory_system.py — Dual-store memory system (SQLite episodic store + ChromaDB semantic index).

Features:
1. SQLite store: exact, lossless persistence of session turns.
2. ChromaDB store: vector index for fast semantic lookup over episode content.
3. Retrieval scoring: Park et al. (2023) composite blending of min-max scaled recency and relevance.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import chromadb
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    from .embedder import MemoryEmbedder
    # pyrefly: ignore [missing-import]
    from .models import EpisodicEntry, RetrievedMemory
except ImportError:
    # pyrefly: ignore [missing-import]
    from embedder import MemoryEmbedder
    # pyrefly: ignore [missing-import]
    from models import EpisodicEntry, RetrievedMemory



DEFAULT_DATA_DIR = Path(__file__).parent / "data"


class MemorySystem:
    """
    Manages persistent episodic memory storage and hybrid recency+relevance retrieval.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        chroma_path: str | Path | None = None,
    ) -> None:
        if db_path is None:
            db_path = DEFAULT_DATA_DIR / "episodic_memory.db"
        if chroma_path is None:
            chroma_path = DEFAULT_DATA_DIR / "chroma_db"

        self.db_path = Path(db_path)
        self.chroma_path = Path(chroma_path)

        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        self._init_sqlite()

        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            metadata={"description": "Semantic embeddings of conversation turns"},
        )

        # Initialize local embedder instance
        self.embedder = MemoryEmbedder.get_instance()

    def _init_sqlite(self) -> None:
        """Create episodic_memory table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score REAL NOT NULL
                );
                """
            )
            conn.commit()

    def add_entry(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
        timestamp: str | None = None,
        importance_score: float | None = None,
    ) -> EpisodicEntry:
        """
        Store a conversation turn into both SQLite (exact record) and ChromaDB (semantic index).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        if importance_score is None:
            # Simple deterministic importance rating based on content length
            importance_score = round(min(1.0, max(0.1, len(content) / 100.0)), 2)

        # Insert into SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO episodic_memory (session_id, timestamp, role, content, importance_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, timestamp, role, content, importance_score),
            )
            conn.commit()
            row_id = cursor.lastrowid

        entry = EpisodicEntry(
            id=row_id,
            session_id=session_id,
            timestamp=timestamp,
            role=role,
            content=content,
            importance_score=importance_score,
        )

        # Embed content and index in ChromaDB using SQLite row_id
        vector = self.embedder.embed([content])[0]
        self.collection.add(
            ids=[str(row_id)],
            embeddings=[vector.tolist()],
            documents=[content],
            metadatas=[
                {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "role": role,
                }
            ],
        )

        return entry

    def get_all_entries(
        self,
        exclude_session_id: str | None = None,
        role: str | None = "user",
    ) -> List[EpisodicEntry]:
        """Fetch candidate entries from SQLite, optionally filtering by session_id and role."""
        query = "SELECT id, session_id, timestamp, role, content, importance_score FROM episodic_memory"
        params: List[str] = []
        conditions: List[str] = []

        if exclude_session_id:
            conditions.append("session_id != ?")
            params.append(exclude_session_id)
        if role:
            conditions.append("role = ?")
            params.append(role)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

        return [
            EpisodicEntry(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                role=row["role"],
                content=row["content"],
                importance_score=row["importance_score"],
            )
            for row in rows
        ]

    def retrieve_memories(
        self,
        query: str,
        current_session_id: str,
        top_k: int = 5,
        now: datetime | None = None,
        role: str | None = "user",
    ) -> List[RetrievedMemory]:
        """
        Retrieve and rank top_k cross-session memories matching the query.

        Candidate pool excludes all entries from current_session_id and defaults to user role entries.
        Scoring formula: score = recency + relevance
        - Recency: exponential decay = 0.995 ** hours_since_creation
        - Relevance: cosine similarity between query and candidate content embeddings
        - Both terms min-max scaled across candidate set to [0, 1] before summing.
        """
        candidates = self.get_all_entries(exclude_session_id=current_session_id, role=role)
        if not candidates:
            return []

        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        query_emb = self.embedder.embed([query])[0]

        # Collect raw metrics for all candidate entries
        raw_recencies: List[float] = []
        raw_relevances: List[float] = []

        # Batch fetch candidate embeddings from ChromaDB
        cand_ids = [str(c.id) for c in candidates]
        chroma_res = self.collection.get(ids=cand_ids, include=["embeddings"])
        embeddings_by_id = {}
        if chroma_res and chroma_res.get("ids"):
            for cid, emb in zip(chroma_res["ids"], chroma_res["embeddings"]):
                embeddings_by_id[int(cid)] = np.array(emb, dtype=np.float32)

        for cand in candidates:
            # Parse timestamp and compute hours since creation
            cand_dt = datetime.fromisoformat(cand.timestamp)
            if cand_dt.tzinfo is None:
                cand_dt = cand_dt.replace(tzinfo=timezone.utc)

            hours_since = max(0.0, (now - cand_dt).total_seconds() / 3600.0)
            raw_rec = 0.995 ** hours_since
            raw_recencies.append(raw_rec)

            # Get embedding for candidate
            if cand.id in embeddings_by_id:
                c_emb = embeddings_by_id[cand.id]
            else:
                c_emb = self.embedder.embed([cand.content])[0]

            raw_rel = self.embedder.cosine_similarity(query_emb, c_emb)
            raw_relevances.append(raw_rel)

        # Min-max scaling with thresholding to avoid noise amplification
        norm_recencies = self._min_max_scale(raw_recencies)
        norm_relevances = self._min_max_scale(raw_relevances)

        results: List[RetrievedMemory] = []
        for i, cand in enumerate(candidates):
            norm_rec = norm_recencies[i]
            norm_rel = norm_relevances[i]
            comp_score = norm_rec + norm_rel

            results.append(
                RetrievedMemory(
                    entry=cand,
                    recency_score=round(norm_rec, 4),
                    relevance_score=round(norm_rel, 4),
                    composite_score=round(comp_score, 4),
                )
            )

        # Sort descending by composite score, breaking ties by recency score then candidate id
        results.sort(key=lambda m: (m.composite_score, m.recency_score, m.entry.id or 0), reverse=True)
        return results[:top_k]

    @staticmethod
    def _min_max_scale(values: List[float], min_diff_threshold: float = 0.01) -> List[float]:
        """
        Min-max scale values to [0, 1].
        If values are virtually identical (range < min_diff_threshold), returns clamped raw values.
        """
        if not values:
            return []
        min_v, max_v = min(values), max(values)
        diff = max_v - min_v
        if diff < min_diff_threshold:
            return [max(0.0, min(1.0, float(v))) for v in values]
        return [float((v - min_v) / diff) for v in values]

