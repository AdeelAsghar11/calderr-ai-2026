"""
graph_builder.py — Knowledge Graph construction, entity deduplication, and Pyvis HTML rendering.

Key functions:
- build_knowledge_graph: Extracts entities/relationships across corpus, merges duplicates, builds NetworkX DiGraph.
- render_pyvis_graph: Renders NetworkX DiGraph into an interactive Pyvis HTML visualization file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
from pyvis.network import Network

try:
    from .extractor import extract_from_paragraph
    from .models import ExtractedEntity, ExtractedRelationship
except ImportError:
    from extractor import extract_from_paragraph
    from models import ExtractedEntity, ExtractedRelationship


def build_knowledge_graph(
    corpus: List[str],
    use_real: bool = False,
) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from corpus paragraphs.

    Performs entity deduplication by merging entity occurrences with the same lower-cased name and type,
    preserving the full set of source paragraph IDs for each node.
    """
    graph = nx.DiGraph()

    all_raw_entities: List[ExtractedEntity] = []
    all_raw_relationships: List[ExtractedRelationship] = []

    # 1. Extract raw entities and relationships across all paragraphs
    for pid, paragraph in enumerate(corpus):
        ents, rels = extract_from_paragraph(paragraph, paragraph_id=pid, use_real=use_real)
        all_raw_entities.extend(ents)
        all_raw_relationships.extend(rels)

    # 2. Merge / Deduplicate entities into canonical nodes
    # Mapping: (normalized_name, entity_type) -> (canonical_name, set_of_paragraph_ids)
    merged_nodes: Dict[Tuple[str, str], Tuple[str, Set[int]]] = {}

    for e in all_raw_entities:
        norm_key = (e.name.strip().lower(), e.entity_type)
        if norm_key not in merged_nodes:
            merged_nodes[norm_key] = (e.name.strip(), {e.source_paragraph_id})
        else:
            canonical_name, pids = merged_nodes[norm_key]
            pids.add(e.source_paragraph_id)

    # Add merged nodes to NetworkX graph
    canonical_map: Dict[str, str] = {}  # norm_name -> canonical_name
    for (norm_name, etype), (canonical_name, pids) in merged_nodes.items():
        canonical_map[norm_name] = canonical_name
        graph.add_node(
            canonical_name,
            entity_type=etype,
            source_paragraph_ids=sorted(list(pids)),
        )

    # Helper function to find canonical node name
    def get_canonical(name: str) -> str:
        name_clean = name.strip().lower()
        for (norm_n, _), (c_name, _) in merged_nodes.items():
            if norm_n == name_clean:
                return c_name
        return name.strip()

    # 3. Add relationships as directed edges
    for rel in all_raw_relationships:
        src = get_canonical(rel.source_entity)
        tgt = get_canonical(rel.target_entity)

        if not graph.has_node(src):
            graph.add_node(src, entity_type="company", source_paragraph_ids=[rel.source_paragraph_id])
        if not graph.has_node(tgt):
            graph.add_node(tgt, entity_type="place", source_paragraph_ids=[rel.source_paragraph_id])

        graph.add_edge(
            src,
            tgt,
            relationship=rel.relationship,
            source_paragraph_id=rel.source_paragraph_id,
        )

    return graph


def render_pyvis_graph(
    graph: nx.DiGraph,
    output_file: str | Path = "graph.html",
) -> Path:
    """
    Render a NetworkX DiGraph into an interactive Pyvis HTML visualization file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="750px", width="100%", directed=True, notebook=False)

    # Color palette for entity types
    color_map = {
        "person": "#97C2FC",    # Soft blue
        "company": "#FFFFB3",   # Soft yellow
        "place": "#FB8072",     # Soft red/coral
    }

    for node, data in graph.nodes(data=True):
        etype = data.get("entity_type", "unknown")
        pids = data.get("source_paragraph_ids", [])
        color = color_map.get(etype, "#DDDDDD")
        title_str = f"<b>{node}</b><br>Type: {etype}<br>Paragraphs: {pids}"

        net.add_node(
            node,
            label=node,
            title=title_str,
            color=color,
            group=etype,
            shape="dot",
            size=25,
        )

    for u, v, data in graph.edges(data=True):
        rel = data.get("relationship", "related")
        pid = data.get("source_paragraph_id", 0)
        edge_title = f"Relationship: {rel} (Paragraph {pid})"

        net.add_edge(
            u,
            v,
            label=rel,
            title=edge_title,
            arrows="to",
        )

    # Save html file
    net.write_html(str(output_path))
    return output_path
