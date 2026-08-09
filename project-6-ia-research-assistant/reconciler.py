"""
reconciler.py — LLM-Powered Fact Extractor and Mem0 Profile Reconciler.

Uses ChatGroq (llama-3.3-70b-versatile) to extract candidate profile facts from session messages,
and applies Mem0 ADD, UPDATE, DELETE, and NOOP reconciliation logic against the user's profile.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from .models import InteractionLog, ProfileFact, ProfileUpdateDecision, UserProfile
except ImportError:
    from models import InteractionLog, ProfileUpdateDecision, UserProfile, ProfileFact

# Load environment variables from .env in repository root
load_dotenv()


class FactExtractor:
    """LLM-based fact extractor using ChatGroq (llama-3.3-70b-versatile)."""

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required. Ensure .env is in the root directory.")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def extract_facts(self, logs: List[InteractionLog]) -> List[ProfileFact]:
        """Extract candidate ProfileFact objects from session user turns using ChatGroq."""
        user_texts = [l.content for l in logs if l.role == "user"]
        if not user_texts:
            return []

        combined_text = "\n".join(user_texts)

        prompt = (
            "You are an expert user profile fact extractor.\n"
            "Analyze the user messages from a research session and extract candidate user profile facts.\n"
            "Valid fields:\n"
            "- 'known_topics': specific research concepts or topics discussed or asked by the user (e.g., 'self-attention', 'multi-head attention', 'positional encoding').\n"
            "- 'preferred_depth': user's stated preference for explanation detail. MUST be strictly 'brief' or 'detailed'.\n"
            "- 'communication_style': stated tone or formatting preferences.\n"
            "- 'open_questions': explicit open questions or follow-up research topics interest.\n\n"
            "Return ONLY a valid JSON list of objects formatted exactly as:\n"
            '[{"field": "known_topics", "content": "self-attention"}, {"field": "preferred_depth", "content": "detailed"}]\n\n'
            f"User Messages:\n{combined_text}\n\n"
            "JSON:"
        )

        res = self.llm.invoke(prompt)
        content = str(res.content).strip()

        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            content = match.group(0)

        try:
            data = json.loads(content)
            facts: List[ProfileFact] = []
            for item in data:
                field = item.get("field")
                val = item.get("content", "").strip()
                if field in ("known_topics", "preferred_depth", "communication_style", "open_questions") and val:
                    facts.append(ProfileFact(field=field, content=val))
            return facts
        except Exception:
            return []


class ProfileReconciler:
    """Mem0 Profile Reconciler managing ADD, UPDATE, DELETE, and NOOP operations."""

    def reconcile(
        self,
        current_profile: UserProfile,
        candidate_facts: List[ProfileFact],
    ) -> Tuple[UserProfile, List[ProfileUpdateDecision]]:
        """
        Reconcile candidate facts against current UserProfile.
        Updates profile fields in-place and returns updated profile + decision logs with written reasoning.
        """
        updated_profile = current_profile.model_copy(deep=True)
        decisions: List[ProfileUpdateDecision] = []
        now_str = datetime.now(timezone.utc).isoformat()

        for fact in candidate_facts:
            decision = self._reconcile_single_fact(updated_profile, fact)
            decisions.append(decision)

        updated_profile.last_updated = now_str
        return updated_profile, decisions

    def _reconcile_single_fact(
        self,
        profile: UserProfile,
        fact: ProfileFact,
    ) -> ProfileUpdateDecision:
        """Process a single candidate fact against profile schema rules."""
        field = fact.field
        content = fact.content.strip()

        # Handle Singleton Field: preferred_depth
        if field == "preferred_depth":
            val = "brief" if "brief" in content.lower() or "concise" in content.lower() or "high-level" in content.lower() else "detailed"
            if profile.preferred_depth != val:
                old_val = profile.preferred_depth
                profile.preferred_depth = val  # type: ignore[assignment]
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="UPDATE",
                    reasoning=f"Updated preferred_depth from '{old_val}' to '{val}' based on new user preference.",
                )
            else:
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="NOOP",
                    reasoning=f"Value '{val}' is already the active preferred_depth.",
                )

        # Handle Singleton Field: communication_style
        elif field == "communication_style":
            if profile.communication_style.lower() != content.lower():
                old_val = profile.communication_style
                profile.communication_style = content
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="UPDATE",
                    reasoning=f"Updated communication_style from '{old_val}' to '{content}'.",
                )
            else:
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="NOOP",
                    reasoning=f"Communication style '{content}' already set.",
                )

        # Handle Collection Field: known_topics
        elif field == "known_topics":
            norm_content = content.lower()
            existing_match = next(
                (t for t in profile.known_topics if t.lower() == norm_content or norm_content in t.lower() or t.lower() in norm_content),
                None,
            )

            if not existing_match:
                profile.known_topics.append(content)
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="ADD",
                    reasoning=f"Added new research topic '{content}' to known_topics collection.",
                )
            else:
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="NOOP",
                    reasoning=f"Topic '{content}' is already present in known_topics collection.",
                )

        # Handle Collection Field: open_questions
        elif field == "open_questions":
            norm_content = content.lower()
            existing_match = next(
                (q for q in profile.open_questions if q.lower() == norm_content), None
            )

            if not existing_match:
                profile.open_questions.append(content)
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="ADD",
                    reasoning=f"Added '{content}' to open_questions collection.",
                )
            else:
                return ProfileUpdateDecision(
                    fact=fact,
                    operation="NOOP",
                    reasoning=f"Question '{content}' is already present in open_questions.",
                )

        return ProfileUpdateDecision(
            fact=fact,
            operation="NOOP",
            reasoning=f"No matching profile field rules for '{field}'.",
        )
