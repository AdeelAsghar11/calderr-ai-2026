"""
models.py — Data models for episodic memory and memory retrieval.

Contains Pydantic definitions for:
- EpisodicEntry: Represents a single raw interaction turn stored in SQLite & ChromaDB.
- RetrievedMemory: Represents a retrieved episodic entry along with its component and composite scores.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class EpisodicEntry(BaseModel):
    """
    Represents an exact, lossless record of a single conversation turn.

    Fields match the SQLite schema and ChromaDB document metadata.
    """
    id: int | None = Field(default=None, description="SQLite autoincrement primary key")
    session_id: str = Field(description="Unique identifier for the conversation session")
    timestamp: str = Field(description="ISO 8601 timestamp string of the interaction")
    role: Literal["user", "assistant"] = Field(description="Speaker role in the interaction")
    content: str = Field(description="Raw text content of the message turn")
    importance_score: float = Field(default=0.5, description="Deterministic importance rating (0.0 to 1.0)")


class RetrievedMemory(BaseModel):
    """
    Represents a candidate memory entry retrieved from past sessions, paired with scoring metrics.

    Scoring uses Park et al. (Generative Agents, 2023) recency + relevance composite blending.
    """
    entry: EpisodicEntry = Field(description="The underlying episodic memory entry")
    recency_score: float = Field(description="Normalized recency score in range [0, 1]")
    relevance_score: float = Field(description="Normalized semantic relevance score in range [0, 1]")
    composite_score: float = Field(description="Composite score = recency_score + relevance_score")
