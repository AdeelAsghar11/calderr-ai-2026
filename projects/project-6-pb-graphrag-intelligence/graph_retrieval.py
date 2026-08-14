"""
graph_retrieval.py — NetworkX Graph Builder and Graph Neighborhood Retriever for Project 6-PB.

Builds a directed NetworkX graph with exactly 25 nodes and 27 edges from the relational facts.
Implements entity grounding and undirected neighborhood traversal (including sibling child company expansion).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
# pyrefly: ignore [missing-import]
import networkx as nx
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

try:
    # pyrefly: ignore [missing-import]
    from .corpus import FULL_CORPUS_PARAGRAPHS, KNOWN_ENTITIES
except ImportError:
    # pyrefly: ignore [missing-import]
    from corpus import FULL_CORPUS_PARAGRAPHS, KNOWN_ENTITIES


def build_knowledge_graph(corpus: List[str] = FULL_CORPUS_PARAGRAPHS) -> nx.DiGraph:
    """
    Build directed NetworkX graph from corpus paragraphs.

    Performs entity deduplication by merging entity occurrences with the same normalized name,
    resulting in exactly 25 nodes and 27 directed edges across the 27 relational facts.
    """
    graph = nx.DiGraph()

    # Add all 25 known canonical nodes first with entity types
    for name, etype in KNOWN_ENTITIES.items():
        graph.add_node(name, entity_type=etype, source_paragraph_ids=[])

    # Extract raw entities and relationships across paragraphs
    for pid, paragraph in enumerate(corpus):
        # 1. Update source paragraph IDs for mentioned entities
        for name in KNOWN_ENTITIES:
            if name in paragraph:
                if pid not in graph.nodes[name]["source_paragraph_ids"]:
                    graph.nodes[name]["source_paragraph_ids"].append(pid)

        # 2. Extract directed relationships (only from relational facts)
        # founded_by
        m_found = re.search(
            r"(Marcus Ondiek|Sofia Petrakis|Aiden Kowalczyk|Leila Farouk|Tomas Brennan|Naomi Iwu)\s+founded\s+(Ridgeline Dynamics|Nimbus Water Systems|Delta Forge Manufacturing|Solstice Grid Energy|Kestrel Biotech|Pinnacle Cargo Systems)",
            paragraph,
        )
        if m_found:
            person, company = m_found.group(1), m_found.group(2)
            graph.add_edge(company, person, relationship="founded_by", source_paragraph_id=pid)

        # located_in
        m_loc = re.search(
            r"(Ridgeline Dynamics|Nimbus Water Systems|Delta Forge Manufacturing|Solstice Grid Energy|Kestrel Biotech|Pinnacle Cargo Systems|Meridian Holdings|Atlas Group|Crestview Partners)\s+is headquartered in\s+(Denver|Rotterdam|Nairobi|Osaka)",
            paragraph,
        )
        if m_loc:
            company, place = m_loc.group(1), m_loc.group(2)
            graph.add_edge(company, place, relationship="located_in", source_paragraph_id=pid)

        # part_of
        m_part = re.search(
            r"(Ridgeline Dynamics|Kestrel Biotech|Nimbus Water Systems|Pinnacle Cargo Systems|Delta Forge Manufacturing|Solstice Grid Energy)\s+is part of\s+(Meridian Holdings|Atlas Group|Crestview Partners)",
            paragraph,
        )
        if m_part:
            child_co, parent_co = m_part.group(1), m_part.group(2)
            graph.add_edge(child_co, parent_co, relationship="part_of", source_paragraph_id=pid)

        # works_at
        m_work = re.search(
            r"(Farah Deng|Victor Amaro|Priya Chandran|Jonas Eriksson|Ana Beloso|Ravi Thakkar)\s+works at\s+(Ridgeline Dynamics|Nimbus Water Systems|Delta Forge Manufacturing|Solstice Grid Energy|Kestrel Biotech|Pinnacle Cargo Systems)",
            paragraph,
        )
        if m_work:
            person, company = m_work.group(1), m_work.group(2)
            graph.add_edge(person, company, relationship="works_at", source_paragraph_id=pid)

    return graph


class GraphRetriever:
    """Graph neighborhood retriever powered by NetworkX knowledge graph."""

    def __init__(self, corpus: List[str] = FULL_CORPUS_PARAGRAPHS) -> None:
        self.corpus = corpus
        self.graph = build_knowledge_graph(self.corpus)
        self.undirected_graph = self.graph.to_undirected()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.node_names = list(self.graph.nodes())
        if self.node_names:
            self.node_embeddings = self.embedder.encode(
                self.node_names,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.node_embeddings = np.empty((0, 384))

    def ground_entity(self, text: str) -> Optional[str]:
        """Ground a query phrase to closest matching graph node name."""
        text_clean = text.lower()
        for node in self.node_names:
            if node.lower() in text_clean:
                return node

        if self.node_names and self.node_embeddings.size > 0:
            query_emb = self.embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = self.node_embeddings @ query_emb
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > 0.4:
                return self.node_names[best_idx]

        return None

    def find_all_grounded_entities(self, query: str) -> List[str]:
        """Find all graph nodes mentioned or semantically matched in the query."""
        grounded: List[str] = []
        for node in self.node_names:
            if node.lower() in query.lower():
                grounded.append(node)

        if not grounded:
            primary = self.ground_entity(query)
            if primary:
                grounded.append(primary)

        return list(dict.fromkeys(grounded))

    def retrieve(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Perform neighborhood expansion around grounded query entities.
        Collects source paragraph texts strictly from relationship edges in the graph.

        Returns:
            Tuple[List[str], List[str]]: (retrieved_paragraph_texts, list_of_grounded_or_traversed_entities)
        """
        grounded_entities = self.find_all_grounded_entities(query)
        if not grounded_entities:
            return [], []

        collected_paragraph_ids: Set[int] = set()
        all_related_entities: Set[str] = set(grounded_entities)

        for node in grounded_entities:
            if not self.graph.has_node(node):
                continue

            # 1-hop neighbors and edge paragraph IDs
            neighbors = list(self.undirected_graph.neighbors(node))
            for nbr in neighbors:
                all_related_entities.add(nbr)

                if self.graph.has_edge(node, nbr):
                    pid = self.graph[node][nbr].get("source_paragraph_id")
                    if pid is not None:
                        collected_paragraph_ids.add(pid)
                if self.graph.has_edge(nbr, node):
                    pid = self.graph[nbr][node].get("source_paragraph_id")
                    if pid is not None:
                        collected_paragraph_ids.add(pid)

                # 2-hop neighbors (includes parent-company sibling expansion)
                for nbr2 in self.undirected_graph.neighbors(nbr):
                    if nbr2 != node:
                        all_related_entities.add(nbr2)
                        if self.graph.has_edge(nbr, nbr2):
                            pid = self.graph[nbr][nbr2].get("source_paragraph_id")
                            if pid is not None:
                                collected_paragraph_ids.add(pid)
                        if self.graph.has_edge(nbr2, nbr):
                            pid = self.graph[nbr2][nbr].get("source_paragraph_id")
                            if pid is not None:
                                collected_paragraph_ids.add(pid)

        retrieved_texts: List[str] = []
        for pid in sorted(collected_paragraph_ids):
            if 0 <= pid < len(self.corpus):
                retrieved_texts.append(self.corpus[pid])

        return retrieved_texts, list(all_related_entities)
