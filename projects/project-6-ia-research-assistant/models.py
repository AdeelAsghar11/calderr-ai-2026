"""
models.py — Strongly typed Pydantic data models for Project 6-I-A Personal Research Assistant.

Defines schemas for episodic interaction logs, user profiles, candidate profile facts,
reconciliation decisions (ADD/UPDATE/DELETE/NOOP), and hybrid retrieved memories.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class InteractionLog(BaseModel):
    """Represents a single conversation turn in the episodic memory store."""

    id: Optional[int] = Field(default=None, description="Auto-incremented primary key in SQLite store.")
    session_id: str = Field(description="Unique identifier for the research session.")
    timestamp: str = Field(description="ISO-8601 timestamp string of the turn.")
    role: Literal["user", "assistant"] = Field(description="Speaker role in the turn.")
    content: str = Field(description="Text message content.")
    importance_score: float = Field(default=1.0, description="Relative importance weighting of the turn.")


class UserProfile(BaseModel):
    """Represents the evolving long-term user profile."""

    known_topics: List[str] = Field(
        default_factory=list,
        description="Collection of research topics previously discussed or learned by the user.",
    )
    preferred_depth: Literal["brief", "detailed"] = Field(
        default="detailed",
        description="Singleton user preference for explanation detail level.",
    )
    communication_style: str = Field(
        default="technical and structured",
        description="Singleton description of user's preferred tone and structure.",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="Collection of active open questions or follow-up topics interest.",
    )
    last_updated: str = Field(
        default="",
        description="ISO-8601 timestamp of last profile reconciliation update.",
    )


class ProfileFact(BaseModel):
    """A candidate fact or preference extracted from user interactions."""

    field: Literal["known_topics", "preferred_depth", "communication_style", "open_questions"] = Field(
        description="Target profile field to update or append."
    )
    content: str = Field(description="The extracted fact string value.")


class ProfileUpdateDecision(BaseModel):
    """Result of reconciling a candidate fact against the existing UserProfile using Mem0 operations."""

    fact: ProfileFact = Field(description="Candidate profile fact evaluated.")
    operation: Literal["ADD", "UPDATE", "DELETE", "NOOP"] = Field(
        description="Reconciliation operation applied."
    )
    reasoning: str = Field(
        description="Written explanation explaining WHY this operation decision was made."
    )


class RetrievedMemory(BaseModel):
    """Scored memory entry retrieved via hybrid recency + relevance search."""

    log: InteractionLog = Field(description="The original episodic interaction turn.")
    recency_score: float = Field(description="Min-max scaled exponential decay recency score.")
    relevance_score: float = Field(description="Min-max scaled embedding cosine similarity relevance score.")
    composite_score: float = Field(description="Combined composite score (recency + relevance).")
