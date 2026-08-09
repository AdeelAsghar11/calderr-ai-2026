"""
dashboard.py — 3-Panel Streamlit Dashboard for Project 6-I-A Personal Research Assistant.

Panels:
1. Current Session: Interactive chat UI connected to ResearchAssistantAgent (ChatGroq LLM).
2. Memory Inspector: SQLite episodic log browser + ChromaDB semantic search over past turns.
3. Profile Viewer: Live UserProfile inspector showing active preferences and reconciliation history.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

PROJ_DIR = Path(__file__).resolve().parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# Load environment variables from .env in repository root
load_dotenv()

try:
    from agent import ResearchAssistantAgent
except ImportError:
    from project_6_ia_research_assistant.agent import ResearchAssistantAgent


def main() -> None:
    st.set_page_config(
        page_title="Personal Research Assistant",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Project 6-I-A: Long-Term Personal Research Assistant")
    st.caption("ChatGroq LLM + SQLite Episodic Memory + ChromaDB Semantic Index + Mem0 Profile Reconciliation")

    # Sidebar Settings
    st.sidebar.header("⚙️ Configuration")
    session_id = st.sidebar.text_input("Active Session ID", value="session_streamlit")

    if st.sidebar.button("Clear Memory & Reset Profile"):
        import shutil
        data_dir = PROJ_DIR / "data"
        if data_dir.exists():
            try:
                shutil.rmtree(data_dir, ignore_errors=True)
            except Exception:
                pass
        st.session_state.pop("agent", None)
        st.success("Cleared all memory & profile state! Re-initialize page.")
        st.rerun()

    # Initialize Agent in Session State (Always Real LLM Mode)
    if "agent" not in st.session_state:
        st.session_state["agent"] = ResearchAssistantAgent()

    agent: ResearchAssistantAgent = st.session_state["agent"]

    # 3-Panel Tabs
    tab_chat, tab_memory, tab_profile = st.tabs(["💬 Current Session", "🔍 Memory Inspector", "👤 Profile Viewer"])

    # -------------------------------------------------------------------------
    # Panel 1: Current Session
    # -------------------------------------------------------------------------
    with tab_chat:
        st.subheader("Interactive Research Session (ChatGroq LLM)")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Render chat messages
        for role, text in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(text)

        # Chat Input
        if prompt := st.chat_input("Ask a research question..."):
            st.session_state["chat_history"].append(("user", prompt))
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving memory & generating ChatGroq LLM response..."):
                    response = agent.generate_response(prompt, session_id=session_id)
                    st.write(response)

                    # Post-session memory write
                    decisions = agent.post_session_memory_write(
                        session_id=session_id,
                        session_turns=[("user", prompt), ("assistant", response)],
                    )

            st.session_state["chat_history"].append(("assistant", response))

            if decisions:
                with st.expander("Live Profile Reconciliation Decisions", expanded=True):
                    for d in decisions:
                        st.info(f"**Field**: `{d.fact.field}` | **Op**: `{d.operation}` | **Content**: `{d.fact.content}`\n\n*Reasoning*: {d.reasoning}")

    # -------------------------------------------------------------------------
    # Panel 2: Memory Inspector
    # -------------------------------------------------------------------------
    with tab_memory:
        st.subheader("Lossless Episodic Store & ChromaDB Semantic Search")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📜 SQLite Episodic Logs")
            logs = agent.episodic_store.get_all_logs()
            if logs:
                log_data = [
                    {
                        "ID": l.id,
                        "Session": l.session_id,
                        "Role": l.role,
                        "Content": l.content,
                        "Timestamp": l.timestamp[:19],
                    }
                    for l in logs
                ]
                st.dataframe(log_data, use_container_width=True)
            else:
                st.info("No episodic logs stored yet.")

        with col2:
            st.markdown("### 🔎 ChromaDB Semantic Search")
            search_query = st.text_input("Semantic search over memory index:", value="attention mechanism")
            if search_query:
                all_logs = agent.episodic_store.get_all_logs(role="user")
                results = agent.semantic_store.retrieve_memories(
                    query=search_query,
                    current_session_id="",
                    candidate_logs=all_logs,
                    top_k=5,
                )
                if results:
                    for r in results:
                        st.write(f"**Turn ID {r.log.id}** (Session: {r.log.session_id})")
                        st.write(f"> {r.log.content}")
                        st.caption(f"Composite Score: {r.composite_score:.3f} (Recency: {r.recency_score:.3f}, Relevance: {r.relevance_score:.3f})")
                        st.divider()
                else:
                    st.info("No semantic memory matches found.")

    # -------------------------------------------------------------------------
    # Panel 3: Profile Viewer
    # -------------------------------------------------------------------------
    with tab_profile:
        st.subheader("Current Reconciled User Profile")

        p = agent.profile
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.metric("Preferred Detail Depth", p.preferred_depth.upper())
            st.write("**Communication Style:**", p.communication_style)
            st.write("**Last Reconciled:**", p.last_updated or "Not updated yet")

        with col_p2:
            st.write("**Known Topics:**")
            if p.known_topics:
                for topic in p.known_topics:
                    st.markdown(f"- `{topic}`")
            else:
                st.info("No topics recorded yet.")

            st.write("**Open Questions:**")
            if p.open_questions:
                for q in p.open_questions:
                    st.markdown(f"- `{q}`")
            else:
                st.info("No open questions recorded.")

        st.divider()
        st.markdown("### 📋 Full Profile JSON")
        st.json(p.model_dump())


if __name__ == "__main__":
    main()
