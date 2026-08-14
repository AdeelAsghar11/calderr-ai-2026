"""
extractor.py — Information extraction module for Knowledge Graph construction.

Supports:
- Stub mode: Rule/pattern-based extraction for offline deterministic parsing of corpus paragraphs.
- Real mode: ChatGroq-based structured extraction for open-ended text generalization.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

try:
    # pyrefly: ignore [missing-import]
    from .models import ExtractedEntity, ExtractedRelationship
except ImportError:
    # pyrefly: ignore [missing-import]
    from models import ExtractedEntity, ExtractedRelationship

# Known entity dictionary for deterministic stub extraction
KNOWN_ENTITIES: dict[str, str] = {
    "Dana Voss": "person",
    "Rina Achebe": "person",
    "Owen Kessler": "person",
    "Priya Nandan": "person",
    "Trailmark Robotics": "company",
    "Glacier Analytics": "company",
    "Vantage Industries": "company",
    "Cobalt Freight": "company",
    "Austin": "place",
    "Toronto": "place",
    "Seattle": "place",
    "Berlin": "place",
}


def extract_from_paragraph(
    paragraph: str,
    paragraph_id: int,
    use_real: bool = False,
) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
    """
    Extract entities and directed relationships from a single text paragraph.
    """
    if use_real:
        return _extract_real(paragraph, paragraph_id)
    return _extract_stub(paragraph, paragraph_id)


def _extract_stub(
    paragraph: str,
    paragraph_id: int,
) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
    """
    Deterministic rule-based extractor for stub mode.
    """
    entities: List[ExtractedEntity] = []
    relationships: List[ExtractedRelationship] = []

    # 1. Identify entity mentions
    found_names: set[str] = set()
    for name, etype in KNOWN_ENTITIES.items():
        if name in paragraph:
            found_names.add(name)
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=etype,  # type: ignore[arg-type]
                    source_paragraph_id=paragraph_id,
                )
            )

    # 2. Extract relationships based on sentence structure
    if "founded" in paragraph:
        # e.g., "Dana Voss founded Trailmark Robotics" or "Rina Achebe founded Glacier Analytics"
        m = re.search(r"(Dana Voss|Rina Achebe)\s+founded\s+(Trailmark Robotics|Glacier Analytics)", paragraph)
        if m:
            person, company = m.group(1), m.group(2)
            relationships.append(
                ExtractedRelationship(
                    source_entity=company,
                    relationship="founded_by",
                    target_entity=person,
                    source_paragraph_id=paragraph_id,
                )
            )

    if "headquartered in" in paragraph:
        # e.g., "Trailmark Robotics is headquartered in Austin"
        m = re.search(
            r"(Trailmark Robotics|Glacier Analytics|Vantage Industries|Cobalt Freight)\s+is headquartered in\s+(Austin|Toronto|Seattle|Berlin)",
            paragraph,
        )
        if m:
            company, place = m.group(1), m.group(2)
            relationships.append(
                ExtractedRelationship(
                    source_entity=company,
                    relationship="located_in",
                    target_entity=place,
                    source_paragraph_id=paragraph_id,
                )
            )

    if "works at" in paragraph:
        # e.g., "Owen Kessler works at Glacier Analytics" or "Priya Nandan works at Cobalt Freight"
        m = re.search(
            r"(Owen Kessler|Priya Nandan)\s+works at\s+(Glacier Analytics|Cobalt Freight)",
            paragraph,
        )
        if m:
            person, company = m.group(1), m.group(2)
            relationships.append(
                ExtractedRelationship(
                    source_entity=person,
                    relationship="works_at",
                    target_entity=company,
                    source_paragraph_id=paragraph_id,
                )
            )

    if "part of" in paragraph:
        # e.g., "Trailmark Robotics is part of Vantage Industries" or "Cobalt Freight is part of Vantage Industries"
        m = re.search(
            r"(Trailmark Robotics|Cobalt Freight)\s+is part of\s+(Vantage Industries)",
            paragraph,
        )
        if m:
            child_co, parent_co = m.group(1), m.group(2)
            relationships.append(
                ExtractedRelationship(
                    source_entity=child_co,
                    relationship="part_of",
                    target_entity=parent_co,
                    source_paragraph_id=paragraph_id,
                )
            )

    return entities, relationships


def _extract_real(
    paragraph: str,
    paragraph_id: int,
) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
    """
    Real LLM-based extraction using ChatGroq for arbitrary text generalization.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required for --real mode.")

    import json
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq

    prompt = (
        "Extract entities and relationships from the following text paragraph.\n"
        "Entities must have types: 'person', 'company', or 'place'.\n"
        "Relationships must use ONLY these predicates: 'works_at', 'founded_by', 'located_in', 'part_of'.\n"
        "Return ONLY a JSON object with schema:\n"
        "{\n"
        "  \"entities\": [{\"name\": \"...\", \"entity_type\": \"person|company|place\"}],\n"
        "  \"relationships\": [{\"source_entity\": \"...\", \"relationship\": \"...\", \"target_entity\": \"...\"}]\n"
        "}\n\n"
        f"Paragraph:\n{paragraph}"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    res = llm.invoke(prompt)
    content = str(res.content).strip()

    # Extract JSON blob
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    data = json.loads(content)

    entities = [
        ExtractedEntity(
            name=e["name"],
            entity_type=e["entity_type"],
            source_paragraph_id=paragraph_id,
        )
        for e in data.get("entities", [])
    ]

    relationships = [
        ExtractedRelationship(
            source_entity=r["source_entity"],
            relationship=r["relationship"],
            target_entity=r["target_entity"],
            source_paragraph_id=paragraph_id,
        )
        for r in data.get("relationships", [])
    ]

    return entities, relationships
