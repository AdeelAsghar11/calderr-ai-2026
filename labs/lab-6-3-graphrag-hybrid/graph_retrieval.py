"""
graph_retrieval.py — Graph retrieval reusing Lab 6.2 graph builder and query agent.

Grounds query entities, expands the graph neighborhood (including 1-hop and 2-hop connected nodes/edges),
and collects all associated source paragraph texts from graph edge relationships.
"""

from __future__ import annotations

import importlib.util
import sys
import typing
from pathlib import Path
from typing import List, Set

import networkx as nx

LAB62_DIR = Path(__file__).resolve().parent.parent / "lab-6-2-knowledge-graph-agent"


def _load_lab62_graph_modules():
    """
    Safely load Lab 6.2 graph builder and query agent.
    Temporarily overrides sys.modules entries so Lab 6.2 internal relative imports resolve correctly,
    then cleans up sys.modules to prevent namespace collisions.
    """
    old_sys_path = list(sys.path)
    old_models_mod = sys.modules.get("models")
    old_extractor_mod = sys.modules.get("extractor")

    try:
        sys.path.insert(0, str(LAB62_DIR))

        # 1. Load lab62 models and rebuild Pydantic models with explicit typing namespace
        spec_m = importlib.util.spec_from_file_location("lab62_models", LAB62_DIR / "models.py")
        if spec_m and spec_m.loader:
            mod_m = importlib.util.module_from_spec(spec_m)
            sys.modules["models"] = mod_m
            spec_m.loader.exec_module(mod_m)

            types_ns = {"Literal": typing.Literal, "TraversalHop": mod_m.TraversalHop}
            mod_m.ExtractedEntity.model_rebuild(_types_namespace=types_ns)
            mod_m.ExtractedRelationship.model_rebuild(_types_namespace=types_ns)
            mod_m.TraversalHop.model_rebuild(_types_namespace=types_ns)
            mod_m.QueryAnswer.model_rebuild(_types_namespace=types_ns)

        # 2. Load lab62 extractor
        spec_ex = importlib.util.spec_from_file_location("lab62_extractor", LAB62_DIR / "extractor.py")
        if spec_ex and spec_ex.loader:
            mod_ex = importlib.util.module_from_spec(spec_ex)
            sys.modules["extractor"] = mod_ex
            spec_ex.loader.exec_module(mod_ex)

        # 3. Load lab62 graph_builder
        spec_gb = importlib.util.spec_from_file_location("lab62_graph_builder", LAB62_DIR / "graph_builder.py")
        if spec_gb is None or spec_gb.loader is None:
            raise ImportError(f"Could not load graph_builder.py from {LAB62_DIR}")
        gb_mod = importlib.util.module_from_spec(spec_gb)
        spec_gb.loader.exec_module(gb_mod)

        # 4. Load lab62 query_agent
        spec_qa = importlib.util.spec_from_file_location("lab62_query_agent", LAB62_DIR / "query_agent.py")
        if spec_qa is None or spec_qa.loader is None:
            raise ImportError(f"Could not load query_agent.py from {LAB62_DIR}")
        qa_mod = importlib.util.module_from_spec(spec_qa)
        spec_qa.loader.exec_module(qa_mod)

        return gb_mod.build_knowledge_graph, qa_mod.KnowledgeGraphQueryAgent
    finally:
        sys.path[:] = old_sys_path
        if old_models_mod is not None:
            sys.modules["models"] = old_models_mod
        else:
            sys.modules.pop("models", None)
        if old_extractor_mod is not None:
            sys.modules["extractor"] = old_extractor_mod
        else:
            sys.modules.pop("extractor", None)


build_knowledge_graph, KnowledgeGraphQueryAgent = _load_lab62_graph_modules()

try:
    # pyrefly: ignore [missing-import]
    from .corpus import FULL_CORPUS_PARAGRAPHS
except ImportError:
    # pyrefly: ignore [missing-import]
    from corpus import FULL_CORPUS_PARAGRAPHS


class GraphRetriever:
    """Graph neighborhood retriever powered by Lab 6.2 Knowledge Graph."""

    def __init__(self, corpus: List[str] = FULL_CORPUS_PARAGRAPHS) -> None:
        self.corpus = corpus
        # Re-use Lab 6.2 build_knowledge_graph logic directly
        self.graph: nx.DiGraph = build_knowledge_graph(self.corpus, use_real=False)
        self.query_agent = KnowledgeGraphQueryAgent(self.graph, self.corpus, use_real=False)

    def find_all_grounded_entities(self, query: str) -> List[str]:
        """
        Identify all graph nodes mentioned or semantically matched in the query.
        """
        grounded: List[str] = []

        # Check all node names for direct substring inclusion first
        for node in self.graph.nodes():
            if node.lower() in query.lower():
                grounded.append(node)

        # Fallback to single primary entity grounding if no substring match
        if not grounded:
            primary = self.query_agent.ground_entity(query)
            if primary:
                grounded.append(primary)

        return list(dict.fromkeys(grounded))  # Preserving order deduplication

    def retrieve(self, query: str) -> tuple[List[str], List[str]]:
        """
        Perform neighborhood expansion around grounded query entities.
        Collects source paragraph texts strictly from edge relationships traversed in the graph.

        Returns:
            tuple[List[str], List[str]]: (retrieved_paragraph_texts, list_of_grounded_or_traversed_entities)
        """
        grounded_entities = self.find_all_grounded_entities(query)
        if not grounded_entities:
            return [], []

        collected_paragraph_ids: Set[int] = set()
        all_related_entities: Set[str] = set(grounded_entities)

        # 1. Inspect paths traced by query agent
        qa_res = self.query_agent.answer_query(query)
        for hop in qa_res.path:
            all_related_entities.add(hop.from_entity)
            all_related_entities.add(hop.to_entity)
            u, v = hop.from_entity, hop.to_entity
            if self.graph.has_edge(u, v):
                pid = self.graph[u][v].get("source_paragraph_id")
                if pid is not None:
                    collected_paragraph_ids.add(pid)
            elif self.graph.has_edge(v, u):
                pid = self.graph[v][u].get("source_paragraph_id")
                if pid is not None:
                    collected_paragraph_ids.add(pid)

        # 2. Perform graph neighborhood expansion (1-hop and 2-hop edges)
        undirected = self.graph.to_undirected()
        for node in grounded_entities:
            if not self.graph.has_node(node):
                continue

            # 1-hop neighbors and edge paragraph IDs
            neighbors = list(undirected.neighbors(node))
            for nbr in neighbors:
                all_related_entities.add(nbr)

                # Directed edge attributes
                if self.graph.has_edge(node, nbr):
                    pid = self.graph[node][nbr].get("source_paragraph_id")
                    if pid is not None:
                        collected_paragraph_ids.add(pid)
                if self.graph.has_edge(nbr, node):
                    pid = self.graph[nbr][node].get("source_paragraph_id")
                    if pid is not None:
                        collected_paragraph_ids.add(pid)

                # 2-hop neighbors for multi-hop graph connectivity
                for nbr2 in undirected.neighbors(nbr):
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

        # Map collected paragraph IDs back to corpus text
        retrieved_texts: List[str] = []
        for pid in sorted(collected_paragraph_ids):
            if 0 <= pid < len(self.corpus):
                retrieved_texts.append(self.corpus[pid])

        return retrieved_texts, list(all_related_entities)
