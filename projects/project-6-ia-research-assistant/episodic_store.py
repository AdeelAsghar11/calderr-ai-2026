"""
episodic_store.py — SQLite-backed lossless episodic memory store.

Persists interaction logs (session_id, timestamp, role, content, importance_score).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

try:
    # pyrefly: ignore [missing-import]
    from .models import InteractionLog
except ImportError:
    # pyrefly: ignore [missing-import]
    from models import InteractionLog

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class EpisodicStore:
    """SQLite persistent store for episodic interaction history."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = DEFAULT_DATA_DIR / "episodic_memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite table schema if not present."""
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

    def add_log(self, log: InteractionLog) -> InteractionLog:
        """Insert a conversation turn into SQLite and return stored record with assigned primary key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO episodic_memory (session_id, timestamp, role, content, importance_score)
                VALUES (?, ?, ?, ?, ?);
                """,
                (log.session_id, log.timestamp, log.role, log.content, log.importance_score),
            )
            conn.commit()
            log_id = cursor.lastrowid

        return InteractionLog(
            id=log_id,
            session_id=log.session_id,
            timestamp=log.timestamp,
            role=log.role,
            content=log.content,
            importance_score=log.importance_score,
        )

    def get_session_logs(self, session_id: str) -> List[InteractionLog]:
        """Fetch all interaction logs for a specific session_id in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, session_id, timestamp, role, content, importance_score FROM episodic_memory WHERE session_id = ? ORDER BY id ASC;",
                (session_id,),
            )
            rows = cursor.fetchall()

        return [
            InteractionLog(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                role=row["role"],
                content=row["content"],
                importance_score=row["importance_score"],
            )
            for row in rows
        ]

    def get_all_logs(
        self,
        exclude_session_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[InteractionLog]:
        """Fetch candidate logs, optionally excluding current session or filtering by role."""
        query = "SELECT id, session_id, timestamp, role, content, importance_score FROM episodic_memory"
        conditions: List[str] = []
        params: List[str] = []

        if exclude_session_id:
            conditions.append("session_id != ?")
            params.append(exclude_session_id)
        if role:
            conditions.append("role = ?")
            params.append(role)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id ASC;"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

        return [
            InteractionLog(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                role=row["role"],
                content=row["content"],
                importance_score=row["importance_score"],
            )
            for row in rows
        ]
