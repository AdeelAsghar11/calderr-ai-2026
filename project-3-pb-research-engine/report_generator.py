"""
report_generator.py — Structured Report Generator

Takes the merged retrieved context and generates a proper research report.

What it does:
  1. Formats all retrieved documents (local + web) into a numbered context block.
     Each source gets a number [1], [2] etc. that the LLM uses for inline citations.
  2. Sends context + query to Groq with a strict reporting prompt.
  3. Groq generates a structured report in Markdown with:
       - A clear title
       - An executive summary (3-5 sentences)
       - Key findings as bullet points
       - Detailed analysis in paragraphs
       - Inline citations like [1] [2] that reference the numbered sources
  4. Parses the output into a ResearchReport Pydantic model.
  5. Appends a full references section from the source metadata.

Why structured output matters here:
  We need the report in a consistent structure so:
  - The Streamlit UI can render each section properly
  - The FastAPI endpoint can return well-typed JSON
  - Citations are always present and traceable
  - 10 demo reports all look consistent

Why inline citations:
  Without citations, the user cannot verify anything the report says.
  With inline citations, every claim points back to a source.
  This is the core value proposition over asking ChatGPT directly —
  the user knows exactly where each fact came from.
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timezone
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

# ── Data models ───────────────────────────────────────────────────────────────
class Citation(BaseModel):
    number:  int
    source:  str           # "local" or "web"
    title:   str
    url:     Optional[str] = None
    snippet: str           # first 200 chars of the chunk


class ResearchReport(BaseModel):
    query:            str
    title:            str
    summary:          str
    key_findings:     list[str]
    detailed_analysis:str
    citations:        list[Citation]
    routing_decision: str
    sources_used:     list[str]    # ["local", "web"] or just one
    timestamp:        str
    report_id:        str


# ── Prompt ────────────────────────────────────────────────────────────────────
REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research analyst producing structured, cited research reports.\n\n"
     "Rules:\n"
     "1. Use ONLY the numbered sources provided in the context\n"
     "2. Every factual claim MUST have an inline citation like [1] or [2]\n"
     "3. Do not add information from outside the provided sources\n"
     "4. If sources conflict, note the disagreement\n"
     "5. Write in clear, professional English\n\n"
     "Output format (use these exact section headers):\n"
     "# [Title]\n\n"
     "## Summary\n[3-5 sentence executive summary]\n\n"
     "## Key Findings\n- [Finding with citation]\n- [Finding with citation]\n\n"
     "## Analysis\n[2-4 paragraphs of detailed analysis with citations]\n\n"
     "## Sources\n[Leave this section empty — it will be filled programmatically]"),
    ("human",
     "Research question: {query}\n\n"
     "Sources:\n{context}\n\n"
     "Write the research report now:"),
])


# ── Generator ─────────────────────────────────────────────────────────────────
class ReportGenerator:
    """
    Converts retrieved documents into a structured research report.

    The core work is context formatting: each document is numbered [1], [2]
    and its source type is labelled so the LLM knows what it is working with.
    The LLM is then instructed to use these numbers for inline citations.
    """

    def __init__(self) -> None:
        self._llm   = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,         # slight creativity for good writing, not for facts
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self._chain = REPORT_PROMPT | self._llm | StrOutputParser()

    def _format_context(self, docs: list[Document]) -> tuple[str, list[Citation]]:
        """
        Format documents into a numbered context block and build citation objects.

        Output looks like:
          [1] (local) Deep learning article
          Backpropagation is an algorithm for computing gradients...

          [2] (web) Backprop explained — https://example.com
          A recent blog post explaining modern approaches...
        """
        context_parts = []
        citations     = []

        for i, doc in enumerate(docs, 1):
            source  = doc.metadata.get("retrieval_source", "unknown")
            title   = doc.metadata.get("citation", doc.metadata.get("title", f"Source {i}"))
            url     = doc.metadata.get("url", None)
            snippet = doc.page_content[:200].replace("\n", " ")

            # Context block seen by the LLM
            source_label = f"({source})" if source in ("local", "web") else ""
            context_parts.append(
                f"[{i}] {source_label} {title}\n"
                f"{doc.page_content[:800]}"
            )

            # Citation object for the report metadata
            citations.append(Citation(
                number  = i,
                source  = source,
                title   = title,
                url     = url,
                snippet = snippet,
            ))

        return "\n\n".join(context_parts), citations

    def _parse_report_sections(self, raw: str) -> dict:
        """
        Extract title, summary, key_findings, and detailed_analysis
        from the LLM's Markdown output.
        """
        lines = raw.strip().split("\n")

        title    = ""
        summary  = ""
        findings = []
        analysis = ""

        current_section = None
        buffer          = []

        for line in lines:
            stripped = line.strip()

            # Section detection
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped.lstrip("# ").strip()
                current_section = "title"
                buffer = []

            elif stripped == "## Summary":
                if current_section == "summary":
                    summary = " ".join(buffer).strip()
                current_section = "summary"
                buffer = []

            elif stripped == "## Key Findings":
                if current_section == "summary":
                    summary = " ".join(buffer).strip()
                current_section = "findings"
                buffer = []

            elif stripped == "## Analysis":
                if current_section == "findings":
                    findings = [b.lstrip("- ").strip() for b in buffer if b.strip().startswith("-")]
                current_section = "analysis"
                buffer = []

            elif stripped.startswith("## "):
                # Any other section (including Sources) — flush analysis
                if current_section == "analysis":
                    analysis = "\n".join(buffer).strip()
                current_section = "other"
                buffer = []

            else:
                if stripped:
                    buffer.append(stripped)

        # Flush the last section
        if current_section == "summary" and not summary:
            summary = " ".join(buffer).strip()
        elif current_section == "findings" and not findings:
            findings = [b.lstrip("- ").strip() for b in buffer if b.strip()]
        elif current_section == "analysis" and not analysis:
            analysis = "\n".join(buffer).strip()

        # Fallback: if parsing failed, use the raw output
        if not title:
            title = "Research Report"
        if not summary:
            summary = raw[:400]
        if not analysis:
            analysis = raw

        return {
            "title":             title,
            "summary":           summary,
            "key_findings":      findings[:8],   # cap at 8 findings
            "detailed_analysis": analysis,
        }

    def generate(
        self,
        query:            str,
        docs:             list[Document],
        routing_decision: str = "both",
        report_id:        str = "",
    ) -> ResearchReport:
        """
        Generate a research report from retrieved documents.

        Args:
            query:            the original research question
            docs:             retrieved documents (local + web mixed)
            routing_decision: which sources were used ("local"/"web"/"both")
            report_id:        unique ID for this report

        Returns:
            ResearchReport with all sections populated and citations included
        """
        if not docs:
            return ResearchReport(
                query=query,
                title="No Results Found",
                summary="No relevant documents were retrieved for this query.",
                key_findings=[],
                detailed_analysis="",
                citations=[],
                routing_decision=routing_decision,
                sources_used=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                report_id=report_id or "empty",
            )

        # Format context and build citation list
        context, citations = self._format_context(docs)

        # Generate the report
        raw_report = self._chain.invoke({
            "query":   query,
            "context": context,
        })

        # Parse sections
        sections = self._parse_report_sections(raw_report)

        # Determine which source types were actually used
        source_types = list({d.metadata.get("retrieval_source", "unknown") for d in docs})

        import uuid
        return ResearchReport(
            query             = query,
            title             = sections["title"],
            summary           = sections["summary"],
            key_findings      = sections["key_findings"],
            detailed_analysis = sections["detailed_analysis"],
            citations         = citations,
            routing_decision  = routing_decision,
            sources_used      = source_types,
            timestamp         = datetime.now(timezone.utc).isoformat(),
            report_id         = report_id or str(uuid.uuid4())[:8],
        )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    from retriever import DualRetriever

    retriever = DualRetriever()
    generator = ReportGenerator()

    query = "What is the attention mechanism in transformers?"
    docs  = retriever.retrieve(query, source="local", top_k=5)
    report = generator.generate(query, docs, routing_decision="local")

    print(f"Title:    {report.title}")
    print(f"Summary:  {report.summary[:200]}...")
    print(f"Findings: {len(report.key_findings)}")
    print(f"Cites:    {len(report.citations)}")
