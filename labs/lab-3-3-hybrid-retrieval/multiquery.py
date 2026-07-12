"""
multiquery.py — Multi-Query Retrieval + Generation  (Lab 3.3 · Thursday)

Problem with single-query retrieval:
  The same question phrased differently retrieves different results.
  Bi-encoders are sensitive to surface form — "how do transformers work?"
  and "explain the transformer architecture" overlap in meaning but
  their embeddings point in slightly different directions.
  The correct chunks may be ranked 2nd for one phrasing and 7th for another.

Solution — Multi-query:
  1. Use an LLM to generate N variations of the original question
  2. Retrieve for each variation (original + N variations)
  3. Take the UNION of all result sets → much larger candidate pool
  4. Deduplicate by content
  5. Re-rank the full pool with a cross-encoder
  6. Return the top-k most relevant results

Why this works:
  Each variation emphasises different semantic aspects of the question.
  "Explain the transformer architecture" pulls structural chunks.
  "How does self-attention compute weights?" pulls formula chunks.
  "What are query, key, and value vectors?" pulls definition chunks.
  The union captures all three. Re-ranking picks the best.

Commands
--------
  python multiquery.py retrieve "how does the transformer attention mechanism work?"
  python multiquery.py ask "what is backpropagation?" --variations 3
  python multiquery.py ask "explain RLHF" --show-variations
"""

from __future__ import annotations

import hashlib
import os
import textwrap
import time
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker

load_dotenv(find_dotenv())

app     = typer.Typer(help="Multi-query retrieval — Lab 3.3", add_completion=False)
console = Console()

# ── Prompts ───────────────────────────────────────────────────────────────────
VARIATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate alternative phrasings of a question to improve document retrieval.\n"
     "Output ONLY the alternative questions, one per line, no numbering, no explanation.\n"
     "Each question must ask for the same information but using different words and emphasis.\n"
     "Vary vocabulary, sentence structure, and which aspect of the topic you emphasise."),
    ("human",
     "Original question: {question}\n\n"
     "Generate {n} alternative phrasings:"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant. Answer using ONLY the provided context.\n"
     "If the context lacks the answer, say: 'The documents do not contain this information.'\n"
     "Be concise and factual. Cite source names when possible."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])


# ── Core class ────────────────────────────────────────────────────────────────
class MultiQueryPipeline:
    """
    Full pipeline: variation generation → retrieval → dedup → rerank → generation.

    The pipeline expands the recall pool by querying from multiple angles,
    then uses a cross-encoder to select the best candidates from that pool.
    """

    def __init__(
        self,
        n_variations:  int  = 3,
        initial_k:     int  = 8,   # per variation
        final_k:       int  = 5,   # after re-ranking
    ) -> None:
        self.n_variations = n_variations
        self.initial_k    = initial_k
        self.final_k      = final_k
        self._retriever   = HybridRetriever()
        self._reranker    = CrossEncoderReranker()
        self._llm         = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.4,          # slight creativity for variation generation
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self._gen_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,             # deterministic for final answer
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate_variations(self, question: str) -> list[str]:
        """Use LLM to generate N rephrased versions of the question."""
        chain     = VARIATION_PROMPT | self._llm | StrOutputParser()
        response  = chain.invoke({"question": question, "n": self.n_variations})
        variations = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return variations[:self.n_variations]

    @staticmethod
    def _doc_hash(doc: Document) -> str:
        """Stable hash of document content for deduplication."""
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    def retrieve(
        self,
        question:   str,
        variations: list[str] | None = None,
    ) -> tuple[list[Document], list[str]]:
        """
        Retrieve and deduplicate across original + all variations.

        Returns (final_ranked_docs, all_queries_used).
        """
        if variations is None:
            variations = self.generate_variations(question)

        all_queries = [question] + variations
        seen_hashes: set[str]   = set()
        candidate_pool: list[Document] = []

        for query in all_queries:
            docs = self._retriever.retrieve(query, top_k=self.initial_k)
            for doc in docs:
                h = self._doc_hash(doc)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    candidate_pool.append(doc)

        # Re-rank the full deduplicated pool against the ORIGINAL question
        ranked = self._reranker.rerank(question, candidate_pool, top_k=self.final_k)
        final_docs = [doc for _, doc in ranked]

        return final_docs, all_queries

    def ask(
        self,
        question:   str,
        variations: list[str] | None = None,
    ) -> tuple[str, list[Document], list[str]]:
        """
        Full pipeline: retrieve → build context → generate answer.
        Returns (answer, retrieved_docs, all_queries).
        """
        docs, queries = self.retrieve(question, variations)

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("topic", doc.metadata.get("source", "unknown"))
            context_parts.append(f"[Source {i} — {source}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        chain  = RAG_PROMPT | self._gen_llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer, docs, queries


# ── Commands ──────────────────────────────────────────────────────────────────
@app.command()
def retrieve(
    question:       str  = typer.Argument(...),
    variations:     int  = typer.Option(3,  "--variations", "-v"),
    initial_k:      int  = typer.Option(8,  "--initial-k"),
    final_k:        int  = typer.Option(5,  "--final-k"),
    show_variations:bool = typer.Option(True, "--show-variations/--hide-variations"),
) -> None:
    """
    Run multi-query retrieval and show the final ranked documents.
    Does NOT call the LLM for answer generation — retrieval only.
    """
    console.print()
    pipeline = MultiQueryPipeline(
        n_variations=variations,
        initial_k=initial_k,
        final_k=final_k,
    )

    # Step 1: generate variations
    console.print(f"[dim]Generating {variations} query variations...[/dim]")
    t0   = time.perf_counter()
    vars_ = pipeline.generate_variations(question)
    t_var = (time.perf_counter() - t0) * 1000

    if show_variations:
        console.print(Panel(
            f"[bold]Original:[/bold] {question}\n\n" +
            "\n".join(f"  [dim]{i+1}.[/dim] {v}" for i, v in enumerate(vars_)),
            title=f"🔄 Query Variations  [{t_var:.0f}ms]",
            border_style="yellow",
        ))

    # Step 2: retrieve + dedup + rerank
    console.print(f"[dim]Retrieving across {variations+1} queries (initial_k={initial_k} each)...[/dim]")
    t0   = time.perf_counter()
    docs, queries = pipeline.retrieve(question, variations=vars_)
    t_ret = (time.perf_counter() - t0) * 1000

    # Show results
    t = Table(
        title=f"Final top-{final_k} after dedup + re-rank  [{t_ret:.0f}ms]",
        border_style="green", show_lines=True, expand=True,
    )
    t.add_column("#",       style="cyan",  width=4)
    t.add_column("Source",  style="cyan",  width=22, no_wrap=True)
    t.add_column("Words",   style="dim",   width=6,  no_wrap=True)
    t.add_column("Snippet", style="white")

    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata or {}
        snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 200, placeholder="…")
        t.add_row(str(i), meta.get("source", "?"), str(meta.get("word_count", "?")), snippet)

    console.print(t)
    console.print(
        f"\n  Pool: {(variations+1) * initial_k} raw → deduped → re-ranked → top {final_k}\n"
    )


@app.command()
def ask(
    question:        str  = typer.Argument(...),
    variations:      int  = typer.Option(3,    "--variations", "-v"),
    initial_k:       int  = typer.Option(8,    "--initial-k"),
    final_k:         int  = typer.Option(5,    "--final-k"),
    show_variations: bool = typer.Option(False, "--show-variations"),
    show_context:    bool = typer.Option(False, "--show-context"),
) -> None:
    """
    Full multi-query RAG: variations → hybrid retrieval → re-rank → generate answer.
    """
    console.print()
    console.print(Panel(
        f"[bold]{question}[/bold]\n"
        f"[dim]variations={variations}  initial_k={initial_k}  final_k={final_k}[/dim]",
        title="🔍 Multi-Query RAG", border_style="blue",
    ))

    pipeline = MultiQueryPipeline(
        n_variations=variations,
        initial_k=initial_k,
        final_k=final_k,
    )

    t0 = time.perf_counter()
    answer, docs, queries = pipeline.ask(question)
    elapsed = (time.perf_counter() - t0) * 1000

    if show_variations:
        console.print(Panel(
            "\n".join(f"  {i}. {q}" for i, q in enumerate(queries)),
            title="Queries used", border_style="dim",
        ))

    if show_context:
        t = Table(title="Retrieved context", border_style="dim", show_lines=True)
        t.add_column("#",  style="cyan", width=4)
        t.add_column("Source", style="cyan", width=20)
        t.add_column("Snippet", style="white")
        for i, doc in enumerate(docs, 1):
            snippet = textwrap.shorten(doc.page_content.replace("\n", " "), 200, placeholder="…")
            t.add_row(str(i), doc.metadata.get("source", "?"), snippet)
        console.print(t)

    console.print(Panel(
        answer,
        title=f"Answer  [{elapsed:.0f}ms total]",
        border_style="green",
    ))
    console.print()


if __name__ == "__main__":
    app()
