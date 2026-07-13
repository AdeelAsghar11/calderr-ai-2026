"""
retriever.py — Dual Retrieval (Local + Web)

Two retrievers, one interface.

LocalRetriever:
  Connects to the ChromaDB we built in lab-3-2.
  Uses hybrid search (BM25 + semantic) from lab-3-3.
  Returns Document objects with metadata showing which source they came from.

WebRetriever:
  Uses DuckDuckGo to search the live internet.
  Fetches the top N search results (title + URL + snippet).
  Wraps each result as a Document so it looks the same as local results.
  No API key required — DuckDuckGo is free and open.

DualRetriever:
  The coordinator. Given a RoutingDecision, it calls whichever
  retrievers are needed. When 'both', it runs them in parallel
  using asyncio so you don't wait for local then web serially.
  The results come back as one merged list tagged with their source.

Why tag with source?
  The report generator needs to know where each piece of information
  came from so it can write proper citations. A local citation looks
  like "[Source: Deep learning article, chunk 12]". A web citation
  looks like "[Source: arxiv.org/abs/2024.xxxxx]".
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document

load_dotenv(find_dotenv())

# ── Paths — reuse lab-3-2 database ───────────────────────────────────────────
CHROMA_PATH     = os.getenv("CHROMA_PATH", "../labs/lab-3-2-rag-pipeline/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wiki_docs_chunk512")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Local Retriever ───────────────────────────────────────────────────────────
class LocalRetriever:
    """
    Retrieves from our ChromaDB knowledge base.

    Uses hybrid search (BM25 + semantic) when lab-3-3 is available,
    falls back to pure semantic search otherwise.

    Each returned Document has metadata["retrieval_source"] = "local"
    and metadata["topic"] showing which Wikipedia article it came from.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef           = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        try:
            self._col = self._client.get_collection(
                name=COLLECTION_NAME, embedding_function=ef
            )
        except Exception:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found at {CHROMA_PATH}.\n"
                f"Run: cd ../labs/lab-3-2-rag-pipeline && python ingest.py run"
            )

        # Try to use hybrid retriever from lab-3-3
        self._hybrid = None
        try:
            sys.path.insert(0, str(Path("../labs/lab-3-3-hybrid-retrieval")))
            # pyrefly: ignore [missing-import]
            from hybrid_retriever import HybridRetriever
            self._hybrid = HybridRetriever(
                chroma_path=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
            )
        except Exception:
            pass  # Fall back to semantic-only

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top_k relevant chunks from ChromaDB."""
        if self._hybrid:
            docs = self._hybrid.retrieve(query, top_k)
        else:
            # Semantic-only fallback
            results = self._col.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas"],
            )
            docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(
                    results["documents"][0], results["metadatas"][0]
                )
            ]

        # Tag every doc so we know it came from local storage
        for doc in docs:
            doc.metadata["retrieval_source"] = "local"
            doc.metadata["citation"] = (
                f"Local knowledge base — {doc.metadata.get('topic', doc.metadata.get('source', 'unknown'))}"
            )
        return docs


# ── Web Retriever ─────────────────────────────────────────────────────────────
class WebRetriever:
    """
    Retrieves from the live internet using DuckDuckGo.

    DuckDuckGo is free and requires no API key. It returns snippets
    (not full page text) which are usually 1-3 sentences summarising
    the page. This is enough for the report generator to cite and
    synthesise from.

    For deeper content, you would swap this with:
    - Tavily API (designed for RAG, returns clean page text)
    - SerpAPI (Google results, requires paid key)
    - Playwright (actually scrapes the page — much slower)
    """

    def __init__(self, max_results: int = 6) -> None:
        self._max_results = max_results

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Search DuckDuckGo and return results as Documents with retry-on-empty/error backoff."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError("Install duckduckgo-search: uv pip install duckduckgo-search")

        raw = []
        retries = 4
        delay = 2.0
        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=min(top_k, self._max_results)))
                if raw:
                    break
                else:
                    print(f"[WebRetriever] Empty results returned for query '{query}' (attempt {attempt + 1}/{retries})")
            except Exception as e:
                print(f"[WebRetriever] DuckDuckGo error (attempt {attempt + 1}/{retries}): {e}")
            
            if attempt < retries - 1:
                import time
                print(f"[WebRetriever] Rate limit / empty results encountered. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2.0

        results = []
        for r in raw:
            # DuckDuckGo returns: {title, href, body}
            # We store the snippet as page_content and the URL in metadata
            doc = Document(
                page_content=f"{r.get('title', '')}\n{r.get('body', '')}",
                metadata={
                    "retrieval_source": "web",
                    "url":              r.get("href", ""),
                    "title":            r.get("title", ""),
                    "citation":         f"{r.get('title', 'Web result')} — {r.get('href', '')}",
                },
            )
            results.append(doc)

        return results


# ── Dual Retriever ────────────────────────────────────────────────────────────
class DualRetriever:
    """
    Coordinates local and web retrieval based on the routing decision.

    When source is 'both', runs local and web retrieval concurrently
    using asyncio.gather so they happen in parallel instead of sequentially.
    This cuts latency roughly in half for the 'both' case.

    Returns a single merged list of Documents tagged with their source.
    The caller (report_generator) uses the tags to write citations.
    """

    def __init__(self) -> None:
        self._local = LocalRetriever()
        self._web   = WebRetriever()

    def retrieve(
        self,
        query:   str,
        source:  str,          # "local" | "web" | "both"
        top_k:   int = 5,
    ) -> list[Document]:
        """
        Route to the right retriever(s) and return merged results.
        Synchronous wrapper around the async implementation.
        """
        return asyncio.run(self._retrieve_async(query, source, top_k))

    async def _retrieve_async(
        self,
        query:  str,
        source: str,
        top_k:  int,
    ) -> list[Document]:
        """
        The actual async retrieval logic.

        'local' — only ChromaDB
        'web'   — only DuckDuckGo
        'both'  — both in parallel via asyncio.gather
        """
        if source == "local":
            return await asyncio.to_thread(self._local.retrieve, query, top_k)

        elif source == "web":
            return await asyncio.to_thread(self._web.retrieve, query, top_k)

        else:  # both
            local_task = asyncio.to_thread(self._local.retrieve, query, top_k)
            web_task   = asyncio.to_thread(self._web.retrieve, query, top_k)

            local_docs, web_docs = await asyncio.gather(local_task, web_task)

            # Interleave: local[0], web[0], local[1], web[1], ...
            # This ensures both sources appear in the top results
            merged = []
            for l, w in zip(local_docs, web_docs):
                merged.extend([l, w])
            # Append any remainder if one list is longer
            merged.extend(local_docs[len(web_docs):])
            merged.extend(web_docs[len(local_docs):])
            return merged


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = DualRetriever()

    print("=== Local retrieval ===")
    docs = retriever.retrieve("what is backpropagation?", source="local", top_k=3)
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.metadata['retrieval_source']}] {d.page_content[:120]}...")

    print("\n=== Web retrieval ===")
    docs = retriever.retrieve("latest AI news 2026", source="web", top_k=3)
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.metadata['retrieval_source']}] {d.metadata.get('title', '')} — {d.metadata.get('url', '')[:60]}")
