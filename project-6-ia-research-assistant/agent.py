"""
agent.py — Research Assistant Agent powered by ChatGroq (llama-3.3-70b-versatile).

Integrates SQLite episodic memory, ChromaDB semantic index, and ChatGroq LLM response generation.
Exhibits personalized behavior adaptation:
1. Short summary acknowledging prior coverage when querying existing known_topics.
2. Strict length reduction when preferred_depth is 'brief' vs 'detailed'.
3. Proactive cross-topic connection referencing prior research topics by name.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from .episodic_store import EpisodicStore
    from .models import InteractionLog, ProfileUpdateDecision, UserProfile
    from .reconciler import FactExtractor, ProfileReconciler
    from .semantic_store import SemanticStore
except ImportError:
    from episodic_store import EpisodicStore
    from models import InteractionLog, ProfileUpdateDecision, UserProfile
    from reconciler import FactExtractor, ProfileReconciler
    from semantic_store import SemanticStore

# Load environment variables from .env in repository root
load_dotenv()

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class ResearchAssistantAgent:
    """Personalized Research Assistant Agent managing long-term profile reconciliation and ChatGroq LLM generation."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required. Ensure .env is in the root directory.")

        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        self.episodic_store = EpisodicStore(db_path=self.data_dir / "episodic_memory.db")
        self.semantic_store = SemanticStore(chroma_path=self.data_dir / "chroma_db")
        self.fact_extractor = FactExtractor()
        self.profile_reconciler = ProfileReconciler()

        self.profile_file = self.data_dir / "user_profile.json"
        self.profile = self._load_profile()

    def _load_profile(self) -> UserProfile:
        """Load persistent UserProfile from disk or initialize default."""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return UserProfile(**data)
            except Exception:
                pass
        return UserProfile()

    def _save_profile(self) -> None:
        """Persist current UserProfile to disk as JSON."""
        with open(self.profile_file, "w", encoding="utf-8") as f:
            json.dump(self.profile.model_dump(), f, indent=2)

    def initialize_session(
        self,
        query: str,
        session_id: str,
    ) -> Tuple[UserProfile, List[str]]:
        """
        Session Initialiser: retrieves cross-session past memories and current UserProfile.
        """
        all_logs = self.episodic_store.get_all_logs(exclude_session_id=session_id, role="user")
        retrieved_memories = self.semantic_store.retrieve_memories(
            query=query,
            current_session_id=session_id,
            candidate_logs=all_logs,
            top_k=3,
        )
        memory_texts = [m.log.content for m in retrieved_memories]
        return self.profile, memory_texts

    def generate_response(
        self,
        query: str,
        session_id: str,
    ) -> str:
        """
        Generate personalized answer adapting to UserProfile and retrieved past memories using ChatGroq.
        """
        profile, past_memories = self.initialize_session(query, session_id)

        memories_str = "\n".join([f"- {m}" for m in past_memories]) if past_memories else "None"
        known_str = ", ".join(profile.known_topics) if profile.known_topics else "None"

        prompt = (
            "You are an intelligent long-term personal research assistant.\n"
            "Respond strictly according to the user profile and past memory context provided below.\n\n"
            f"User Profile:\n"
            f"- Preferred Depth: {profile.preferred_depth}\n"
            f"- Known Topics Previously Covered: [{known_str}]\n"
            f"- Communication Style: {profile.communication_style}\n\n"
            f"Relevant Past Cross-Session Memories:\n{memories_str}\n\n"
            f"CRITICAL BEHAVIOR GUIDELINES:\n"
            f"1. DETAIL DEPTH CONTROL: If preferred_depth is 'brief', generate a CONCISE answer (2-4 sentences max). Do NOT write long sections or detailed bullet points when preferred_depth is 'brief'. If preferred_depth is 'detailed', provide an extensive, detailed technical breakdown.\n"
            f"2. PRIOR COVERAGE ACKNOWLEDGMENT: If the question asks about a topic already in Known Topics (such as 'self-attention'), explicitly state that this topic was previously covered in Session 1 / earlier research, and provide a brief summary instead of a full re-explanation.\n"
            f"3. PROACTIVE CROSS-TOPIC CONNECTION: If the question asks how a new concept relates to earlier topics, explicitly mention and connect it by name to ALL relevant prior topics stored in Known Topics (specifically both 'self-attention' and 'multi-head attention').\n\n"
            f"User Question: {query}\n\n"
            "Response:"
        )

        res = self.llm.invoke(prompt)
        return str(res.content).strip()

    def post_session_memory_write(
        self,
        session_id: str,
        session_turns: List[Tuple[str, str]],
    ) -> List[ProfileUpdateDecision]:
        """
        Post-Session Memory Writer:
        1. Logs turns to SQLite episodic store & ChromaDB semantic index.
        2. Extracts candidate facts using ChatGroq.
        3. Reconciles facts against UserProfile using ADD/UPDATE/DELETE/NOOP.
        4. Saves updated UserProfile to disk.
        """
        created_logs: List[InteractionLog] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for role, text in session_turns:
            log = InteractionLog(
                session_id=session_id,
                timestamp=now_iso,
                role=role,  # type: ignore[arg-type]
                content=text,
                importance_score=1.0,
            )
            stored_log = self.episodic_store.add_log(log)
            self.semantic_store.add_log_embedding(stored_log)
            created_logs.append(stored_log)

        # Fact Extraction + Profile Reconciliation
        candidate_facts = self.fact_extractor.extract_facts(created_logs)
        updated_profile, decisions = self.profile_reconciler.reconcile(
            self.profile, candidate_facts
        )

        self.profile = updated_profile
        self._save_profile()
        return decisions
