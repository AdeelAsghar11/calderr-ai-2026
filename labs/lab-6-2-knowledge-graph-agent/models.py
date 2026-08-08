"""
models.py — Typed Pydantic data models for Lab 6.2 Knowledge Graph Agent.

Models:
- ExtractedEntity: Single entity extracted from a corpus paragraph.
- ExtractedRelationship: Directed relationship tuple between two entities.
- TraversalHop: Individual hop along a graph reasoning path with stored direction.
- QueryAnswer: Complete query execution result with reasoning trace and baseline comparison.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """Represents a named entity extracted from text."""

    name: str = Field(description="Canonical or raw name of the entity")
    entity_type: Literal["person", "company", "place"] = Field(description="Category type of the entity")
    source_paragraph_id: int = Field(description="0-indexed paragraph ID where entity was extracted")


class ExtractedRelationship(BaseModel):
    """Represents a directed relationship between two entities."""

    source_entity: str = Field(description="Source entity name")
    relationship: Literal["works_at", "founded_by", "located_in", "part_of"] = Field(
        description="Relationship predicate type from fixed vocabulary"
    )
    target_entity: str = Field(description="Target entity name")
    source_paragraph_id: int = Field(description="0-indexed paragraph ID where relationship was extracted")


class TraversalHop(BaseModel):
    """Represents a single hop along an undirected graph path, preserving directed schema info."""

    from_entity: str = Field(description="Starting entity node for this hop")
    relationship: str = Field(description="Relationship label on the edge")
    to_entity: str = Field(description="Destination entity node for this hop")
    direction: Literal["forward", "reverse"] = Field(
        description="Direction traversed relative to stored directed edge orientation"
    )


class QueryAnswer(BaseModel):
    """Represents the final structured answer to a multi-hop query."""

    question: str = Field(description="Original natural language query question")
    grounded_entities: list[str] = Field(description="List of graph entity nodes grounded from question")
    path: list[TraversalHop] = Field(description="Sequence of traversal hops representing graph reasoning path")
    answer: str = Field(description="Final answer text or entity name")
    keyword_search_would_succeed: bool = Field(
        description="True if a single corpus paragraph contains both query subject and answer entities"
    )
