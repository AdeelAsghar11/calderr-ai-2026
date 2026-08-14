"""
generate_qa_examples.py — Generate the 15 Q&A Examples Deliverable

Uses hybrid retrieval (BM25 + semantic, RRF fusion) — the version that
fixed the retrieval misses found in the first run: RAGAS metrics not
found despite existing verbatim in CalderR_Week-3.pdf, HACKDATA judges'
feedback retrieved from the wrong chunk, and BSL preprocessing technique
not connected across two documents.

The questions span every document type in the knowledge base: resume/
portfolio, GitHub project READMEs, all 4 CalderR internship weeks, and
both hackathon writeups — so the transcript demonstrates retrieval
working correctly across the whole corpus, not just one file.

Run:
  python generate_qa_examples.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rich.console import Console

# pyrefly: ignore [missing-import]
from hybrid import PersonalHybridRetriever
# pyrefly: ignore [missing-import]
from kb import RAG_PROMPT, build_context, DEFAULT_TOP_K

load_dotenv(find_dotenv())
console = Console()

# ── 15 questions spanning the whole knowledge base ────────────────────────────
QUESTIONS = [
    "What is my current CGPA and academic standing?",
    "What did I build for the HACKDATA V1 hackathon and what problem did it solve?",
    "What accuracy did my BSL hand gesture recognition project achieve?",
    "What tech stack did I use for ThreatLens?",
    "What topics were covered in Week 1 of the CalderR internship?",
    "What is the RAGAS framework and which metrics does it include?",
    "What was my role in the DevOps internship at SPS?",
    "What certifications have I completed from DeepLearning.AI?",
    "What did the judges say about my HACKDATA V1 project?",
    "What was the outcome of the HITEC Mega Code War AI Model Training competition?",
    "What is GDG On Campus COMSATS Wah and what is my role there?",
    "What preprocessing technique improved my BSL model's performance?",
    "What is the architecture of the DermVision application?",
    "What did Week 2 of the CalderR internship cover regarding tool calling?",
    "What feedback did I receive at the Pitch Perfect competition?",
]


def main() -> None:
    console.print(f"\n[bold]Generating {len(QUESTIONS)} Q&A examples (hybrid retrieval)...[/bold]\n")

    retriever = PersonalHybridRetriever()   # loaded once, reused across all 15 questions
    llm    = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    chain  = RAG_PROMPT | llm | StrOutputParser()

    lines = [
        "# Personal Knowledge Base — 15 Q&A Examples\n",
        "Retrieval: hybrid (BM25 keyword + semantic, RRF fusion), top_k=8.\n",
        "Each entry shows the question, the source documents retrieved by "
        "the RAG pipeline, and the generated answer grounded in those sources.\n",
    ]

    for i, q in enumerate(QUESTIONS, 1):
        console.print(f"  [{i:02d}/15] {q[:55]}...")

        docs    = retriever.retrieve(q, top_k=DEFAULT_TOP_K)
        context = build_context(docs)
        answer  = chain.invoke({"context": context, "question": q})
        sources = sorted({d.metadata.get("filename", "?") for d in docs})

        lines.append(f"## Q{i}: {q}\n")
        lines.append(f"**Sources retrieved:** {', '.join(sources)}\n")
        lines.append(f"**Answer:**\n\n{answer}\n")
        lines.append("---\n")

        time.sleep(0.3)  # avoid rate limiting

    out_path = Path("qa_examples.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]✓ Saved {len(QUESTIONS)} Q&A examples → {out_path}[/green]\n")


if __name__ == "__main__":
    main()