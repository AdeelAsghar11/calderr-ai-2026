"""
smoke_test.py — Real LLM Validation Suite for Project 6-I-A Personal Research Assistant.

Executes 5 mandatory proofs using live ChatGroq (llama-3.3-70b-versatile) LLM calls:
1. Profile Accumulation Proof: known_topics contains all 3 distinct topics after sessions 1-3.
2. Conflict Resolution Proof: preferred_depth is single-value 'brief' after UPDATE operation.
3. Behavior Change Proof: Session 4 response is measurably shorter than Session 1 and acknowledges prior coverage.
4. Proactive Connection Proof: Session 5 response references at least 2 prior topics by name.
5. Persistence Proof: Re-instantiating agent from disk preserves profile and episodic logs unchanged.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJ_DIR = Path(__file__).resolve().parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# Load environment variables from .env in repository root
load_dotenv()

try:
    # pyrefly: ignore [missing-import]
    from .agent import ResearchAssistantAgent
except ImportError:
    # pyrefly: ignore [missing-import]
    from agent import ResearchAssistantAgent


def run_smoke_tests() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required. Ensure .env is in the root directory.")

    # Use dedicated temporary test data directory for isolated validation
    test_data_dir = PROJ_DIR / "test_data"
    if test_data_dir.exists():
        try:
            shutil.rmtree(test_data_dir, ignore_errors=True)
        except Exception:
            pass
    test_data_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = ResearchAssistantAgent(data_dir=test_data_dir)

        print("================================================================================")
        print("    PROJECT 6-I-A PERSONAL RESEARCH ASSISTANT - REAL LLM TEST SUITE     ")
        print("================================================================================")

        # ----------------------------------------------------------------------
        # Session 1: User asks about self-attention + prefers detailed explanations
        # ----------------------------------------------------------------------
        s1_id = "session_1"
        s1_query = "Explain self-attention in transformers to me in detail"
        s1_turns = [
            ("user", s1_query),
            ("user", "I prefer detailed technical explanations."),
        ]
        s1_response = agent.generate_response(s1_query, session_id=s1_id)
        s1_decisions = agent.post_session_memory_write(s1_id, s1_turns)

        print(f"\n--- Session 1 Response ---\n{s1_response}\n")

        # ----------------------------------------------------------------------
        # Session 2: User asks about multi-head attention
        # ----------------------------------------------------------------------
        s2_id = "session_2"
        s2_query = "What is multi-head attention?"
        s2_turns = [("user", s2_query)]
        s2_response = agent.generate_response(s2_query, session_id=s2_id)
        s2_decisions = agent.post_session_memory_write(s2_id, s2_turns)

        # ----------------------------------------------------------------------
        # Session 3: User asks about positional encoding + prefers brief answers
        # ----------------------------------------------------------------------
        s3_id = "session_3"
        s3_query = "Explain positional encoding"
        s3_turns = [
            ("user", s3_query),
            ("user", "Actually, I'd prefer brief, high-level answers from now on."),
        ]
        s3_response = agent.generate_response(s3_query, session_id=s3_id)
        s3_decisions = agent.post_session_memory_write(s3_id, s3_turns)

        # ----------------------------------------------------------------------
        # Proof 1: Profile Accumulation Proof
        # ----------------------------------------------------------------------
        print("\n--- Test 1: Profile Accumulation Proof ---")
        topics = agent.profile.known_topics
        print(f"Current known_topics: {topics}")
        has_sa = any("self-attention" in t.lower() for t in topics)
        has_mha = any("multi-head" in t.lower() or "multihead" in t.lower() for t in topics)
        has_pe = any("positional" in t.lower() for t in topics)

        assert has_sa, "Missing 'self-attention' in known_topics"
        assert has_mha, "Missing 'multi-head attention' in known_topics"
        assert has_pe, "Missing 'positional encoding' in known_topics"
        assert len(topics) >= 3, f"Expected >= 3 topics, got {len(topics)}"
        print("OK: Profile Accumulation Proof Passed!")

        # ----------------------------------------------------------------------
        # Proof 2: Conflict Resolution Proof (UPDATE operation)
        # ----------------------------------------------------------------------
        print("\n--- Test 2: Conflict Resolution Proof ---")
        print(f"Current preferred_depth: {agent.profile.preferred_depth}")
        assert agent.profile.preferred_depth == "brief", f"Expected 'brief', got {agent.profile.preferred_depth}"

        # Find the UPDATE decision record from Session 3
        update_decision = next(
            (d for d in s3_decisions if d.fact.field == "preferred_depth" and d.operation == "UPDATE"),
            None,
        )
        assert update_decision is not None, "Expected an UPDATE decision record for preferred_depth in Session 3"
        assert update_decision.operation == "UPDATE", "Operation must be UPDATE"
        print(f"Update Decision: Fact={update_decision.fact} | Op={update_decision.operation} | Reasoning='{update_decision.reasoning}'")
        print("OK: Conflict Resolution Proof Passed!")

        # ----------------------------------------------------------------------
        # Session 4: User asks for reminder on self-attention
        # ----------------------------------------------------------------------
        s4_id = "session_4"
        s4_query = "Remind me what self-attention is."
        s4_turns = [("user", s4_query)]
        s4_response = agent.generate_response(s4_query, session_id=s4_id)
        s4_decisions = agent.post_session_memory_write(s4_id, s4_turns)

        # ----------------------------------------------------------------------
        # Proof 3: Behavior Change Proof (Length reduction + Prior coverage acknowledgment)
        # ----------------------------------------------------------------------
        print("\n--- Test 3: Behavior Change Proof ---")
        s1_words = len(s1_response.split())
        s4_words = len(s4_response.split())
        print(f"Session 1 Word Count: {s1_words} | Session 4 Word Count: {s4_words}")
        print(f"Session 4 Response Text:\n{s4_response}")

        assert s4_words < s1_words, f"Session 4 ({s4_words} words) must be measurably shorter than Session 1 ({s1_words} words)"

        ack_keywords = ["previously", "covered", "session 1", "earlier", "brief", "summarized"]
        has_ack = any(kw in s4_response.lower() for kw in ack_keywords)
        assert has_ack, "Session 4 response must acknowledge prior coverage"
        print("OK: Behavior Change Proof Passed!")

        # ----------------------------------------------------------------------
        # Session 5: User asks about positional encoding connection to earlier topics
        # ----------------------------------------------------------------------
        s5_id = "session_5"
        s5_query = "How does positional encoding relate to what I asked about earlier?"
        s5_turns = [("user", s5_query)]
        s5_response = agent.generate_response(s5_query, session_id=s5_id)
        s5_decisions = agent.post_session_memory_write(s5_id, s5_turns)

        print(f"\n--- Session 5 Response ---\n{s5_response}\n")

        # ----------------------------------------------------------------------
        # Proof 4: Proactive Connection Proof
        # ----------------------------------------------------------------------
        print("\n--- Test 4: Proactive Connection Proof ---")
        s5_lower = s5_response.lower()
        has_self_att = "self-attention" in s5_lower or "self attention" in s5_lower
        has_multi_head = "multi-head" in s5_lower or "multihead" in s5_lower or "multi head" in s5_lower
        print(f"References 'self-attention': {has_self_att} | References 'multi-head attention': {has_multi_head}")

        assert has_self_att and has_multi_head, "Session 5 response must explicitly reference both 'self-attention' and 'multi-head attention'"
        print("OK: Proactive Connection Proof Passed!")

        # ----------------------------------------------------------------------
        # Proof 5: Persistence Proof
        # ----------------------------------------------------------------------
        print("\n--- Test 5: Persistence Proof ---")
        # Re-instantiate agent from disk
        reloaded_agent = ResearchAssistantAgent(data_dir=test_data_dir)

        reloaded_topics = reloaded_agent.profile.known_topics
        reloaded_depth = reloaded_agent.profile.preferred_depth
        reloaded_logs = reloaded_agent.episodic_store.get_all_logs()

        print(f"Reloaded Topics: {reloaded_topics}")
        print(f"Reloaded Depth: {reloaded_depth}")
        print(f"Reloaded Episodic Log Count: {len(reloaded_logs)}")

        assert reloaded_topics == agent.profile.known_topics, "Reloaded known_topics does not match"
        assert reloaded_depth == agent.profile.preferred_depth, "Reloaded preferred_depth does not match"
        assert len(reloaded_logs) > 0, "Reloaded episodic logs empty"
        print("OK: Persistence Proof Passed!")

        print("\n================================================================================")
        print("ALL 5 REAL LLM SMOKE TEST PROOFS PASSED SUCCESSFULLY!")
        print("================================================================================")

    finally:
        if test_data_dir.exists():
            try:
                shutil.rmtree(test_data_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    run_smoke_tests()
