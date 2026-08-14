"""
app.py — Streamlit Frontend

The user-facing UI. Runs separately from FastAPI.

Layout:
  Sidebar  — recent reports history, source mode selector
  Main     — search bar, routing decision badge, report display

What happens when you submit a query:
  1. Calls the same Python functions as api.py (direct import, not HTTP)
     This avoids needing FastAPI running just to use the UI.
  2. Shows which source was selected and why (routing decision)
  3. Streams the report sections as they become available
  4. Renders citations as a numbered reference list at the bottom
  5. Saves the report to reports/ for the history sidebar

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Research Engine",
    page_icon  = "🔬",
    layout     = "wide",
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ── Component loading (cached so models load once per session) ─────────────────
@st.cache_resource(show_spinner="Loading research engine components...")
def load_components():
    # pyrefly: ignore [missing-import]
    from router import KnowledgeRouter
    # pyrefly: ignore [missing-import]
    from retriever import DualRetriever
    # pyrefly: ignore [missing-import]
    from report_generator import ReportGenerator
    return KnowledgeRouter(), DualRetriever(), ReportGenerator()


# ── Helpers ───────────────────────────────────────────────────────────────────
def source_badge(source: str) -> str:
    colours = {"local": "🟢", "web": "🌐", "both": "⚡"}
    return colours.get(source, "❓")


def render_report(report) -> None:
    """Render a ResearchReport in the main area."""

    # Title + metadata
    st.title(report.title)
    col1, col2, col3 = st.columns(3)
    col1.metric("Sources", " + ".join(report.sources_used) or "none")
    col2.metric("Citations", len(report.citations))
    col3.metric("Findings", len(report.key_findings))
    st.divider()

    # Summary
    st.subheader("Summary")
    st.write(report.summary)

    # Key findings
    if report.key_findings:
        st.subheader("Key Findings")
        for finding in report.key_findings:
            st.markdown(f"- {finding}")

    # Detailed analysis
    if report.detailed_analysis:
        st.subheader("Analysis")
        st.markdown(report.detailed_analysis)

    # Citations
    if report.citations:
        st.divider()
        st.subheader("References")
        for cite in report.citations:
            icon = "📄" if cite.source == "local" else "🌐"
            if cite.url:
                st.markdown(f"[{cite.number}] {icon} [{cite.title}]({cite.url})")
            else:
                st.markdown(f"[{cite.number}] {icon} {cite.title}")
            with st.expander("Preview", expanded=False):
                st.caption(cite.snippet)


def save_report(report) -> None:
    path = REPORTS_DIR / f"{report.report_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_recent_reports(n: int = 10) -> list[dict]:
    reports = []
    for path in sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:n]:
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return reports


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Engine")
    st.caption("Local knowledge base + live web search")
    st.divider()

    source_mode = st.radio(
        "Source mode",
        options=["auto", "local", "web", "both"],
        index=0,
        help=(
            "auto  — LLM decides which source to use\n"
            "local — ChromaDB only (AI/ML topics)\n"
            "web   — live internet only\n"
            "both  — always use both"
        ),
    )

    top_k = st.slider("Results per source", min_value=3, max_value=10, value=5)

    st.divider()
    st.subheader("Recent Reports")
    recent = load_recent_reports()
    if recent:
        for r in recent[:8]:
            if st.button(
                r.get("title", r.get("query", ""))[:40],
                key=r.get("report_id", ""),
                use_container_width=True,
            ):
                st.session_state["loaded_report"] = r
    else:
        st.caption("No reports yet. Run a search.")


# ── Main area ──────────────────────────────────────────────────────────────────
st.header("Research Query")

query = st.text_input(
    "What do you want to research?",
    placeholder="e.g. How does the transformer attention mechanism work?",
)

col_search, col_clear = st.columns([1, 4])
search_clicked = col_search.button("🔍 Research", type="primary", use_container_width=True)

# Load a historical report from sidebar click
if "loaded_report" in st.session_state:
    # pyrefly: ignore [missing-import]
    from report_generator import ResearchReport
    try:
        report = ResearchReport.model_validate(st.session_state.pop("loaded_report"))
        st.info(f"Loaded saved report: **{report.title}**")
        render_report(report)
    except Exception as e:
        st.error(f"Failed to load report: {e}")
    st.stop()

# Run a new research query
if search_clicked and query.strip():
    try:
        router, retriever, generator = load_components()
    except Exception as e:
        st.error(f"Failed to load components: {e}\n\nMake sure ChromaDB is set up (lab-3-2).")
        st.stop()

    # ── Route ─────────────────────────────────────────────────────────────────
    with st.spinner("Routing query..."):
        if source_mode == "auto":
            decision = router.route(query)
            source   = decision.source.value
            reason   = decision.reasoning
        else:
            source = source_mode
            reason = f"Manual override: {source}"

    # Show routing decision
    badge = source_badge(source)
    st.info(f"{badge} **Routing → {source.upper()}** — {reason}")

    # ── Retrieve ──────────────────────────────────────────────────────────────
    with st.spinner(f"Retrieving from {source}..."):
        t0   = time.perf_counter()
        docs = retriever.retrieve(query, source=source, top_k=top_k)
        t_retrieve = (time.perf_counter() - t0) * 1000

    if not docs:
        st.warning("No documents retrieved. Try a different query or source mode.")
        st.stop()

    st.caption(f"Retrieved {len(docs)} sources in {t_retrieve:.0f}ms")

    # ── Generate ──────────────────────────────────────────────────────────────
    with st.spinner("Generating research report..."):
        import uuid
        t0     = time.perf_counter()
        report = generator.generate(
            query            = query,
            docs             = docs,
            routing_decision = source,
            report_id        = str(uuid.uuid4())[:8],
        )
        t_gen = (time.perf_counter() - t0) * 1000

    st.caption(f"Generated in {t_gen:.0f}ms · Total: {t_retrieve + t_gen:.0f}ms")
    st.divider()

    # ── Render ────────────────────────────────────────────────────────────────
    render_report(report)

    # Save + offer download
    save_report(report)
    st.download_button(
        "⬇ Download report (JSON)",
        data     = report.model_dump_json(indent=2),
        file_name= f"report_{report.report_id}.json",
        mime     = "application/json",
    )

elif search_clicked:
    st.warning("Please enter a research question.")

else:
    st.markdown("""
    **How to use:**
    1. Type a research question in the box above
    2. Choose source mode in the sidebar (auto = recommended)
    3. Click **Research**

    **What this does:**
    - Routes your query to the right knowledge source
    - Searches your local AI/ML knowledge base and/or live web
    - Generates a structured research report with inline citations

    **Example queries to try:**
    - `What is the attention mechanism in transformers?` → local
    - `What are the latest AI developments in 2026?` → web
    - `How does RAG compare to current GPT-4 performance?` → both
    """)
