"""
smoke_test.py — Automated verification suite for Lab 6.1 Memory-Augmented Chatbot.

Executes all 4 required proof cases:
1. Retrieval-correctness proof (3-session validation scenario: shellfish allergy recall from Session 1).
2. Cross-session exclusion proof (asserts current session entries are excluded from retrieval).
3. Injection proof (asserts top retrieved memory content is injected into response).
4. Scoring sanity check (asserts more recent memory ranks higher given identical relevance).
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    # pyrefly: ignore [missing-import]
    from .chatbot import make_stub_chatbot
    # pyrefly: ignore [missing-import]
    from .memory_system import MemorySystem
except ImportError:
    # pyrefly: ignore [missing-import]
    from chatbot import make_stub_chatbot
    # pyrefly: ignore [missing-import]
    from memory_system import MemorySystem


def run_smoke_tests() -> None:
    print("=" * 70)
    print("RUNNING LAB 6.1 MEMORY CHATBOT SMOKE TESTS")
    print("=" * 70)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:

        temp_path = Path(temp_dir)
        db_path = temp_path / "test_episodic.db"
        chroma_path = temp_path / "test_chroma"

        chatbot = make_stub_chatbot(db_path=db_path, chroma_path=chroma_path)

        # -------------------------------------------------------------------------
        # CASE 1: The Retrieval-Correctness Proof & CASE 3: The Injection Proof
        # -------------------------------------------------------------------------
        print("\n[CASE 1 & 3] Running 3-session validation scenario...")

        session_1 = "session_1_allergy"
        session_2 = "session_2_coding"
        session_3 = "session_3_query"

        # Session 1: User discloses allergy + unrelated turn
        now_utc = datetime.now(timezone.utc)
        t1_now = (now_utc - timedelta(minutes=10)).isoformat()
        s1_reply1, _ = chatbot.process_user_turn(
            session_id=session_1,
            user_message="I'm allergic to shellfish, especially shrimp.",
            timestamp=t1_now,
        )
        s1_reply2, _ = chatbot.process_user_turn(
            session_id=session_1,
            user_message="I also enjoy mountain hiking on weekends.",
            timestamp=t1_now,
        )

        # Session 2: Entirely unrelated coding conversation
        t2_now = (now_utc - timedelta(minutes=5)).isoformat()
        s2_reply1, _ = chatbot.process_user_turn(
            session_id=session_2,
            user_message="Can you help me write a Python script for sorting files?",
            timestamp=t2_now,
        )
        s2_reply2, _ = chatbot.process_user_turn(
            session_id=session_2,
            user_message="I prefer spaces over tabs for code indentation.",
            timestamp=t2_now,
        )


        # Session 3: Fresh session query requiring recall of Session 1
        t3_query = "What should I avoid ordering at a seafood restaurant?"
        t3_reply, retrieved_s3 = chatbot.process_user_turn(
            session_id=session_3,
            user_message=t3_query,
        )

        print(f"Session 3 Query: {t3_query!r}")
        print(f"Retrieved memories count: {len(retrieved_s3)}")
        for idx, mem in enumerate(retrieved_s3, start=1):
            print(
                f"  [{idx}] Session: {mem.entry.session_id} | "
                f"Content: {mem.entry.content!r} | "
                f"Recency: {mem.recency_score:.4f} | "
                f"Relevance: {mem.relevance_score:.4f} | "
                f"Composite: {mem.composite_score:.4f}"
            )

        # CASE 1 ASSERTS
        assert len(retrieved_s3) > 0, "Retrieval returned empty results for Session 3 query!"
        top_memory = retrieved_s3[0]
        assert top_memory.entry.session_id == session_1, (
            f"Case 1 Failure: Top retrieved session was '{top_memory.entry.session_id}', expected '{session_1}'"
        )
        assert "shellfish" in top_memory.entry.content.lower(), (
            f"Case 1 Failure: Top retrieved content was {top_memory.entry.content!r}, expected shellfish allergy"
        )
        print(
            f"[OK] CASE 1 PASSED: Top retrieved memory is Session 1 shellfish allergy "
            f"(Composite Score: {top_memory.composite_score:.4f})"
        )

        # CASE 3 ASSERTS
        assert top_memory.entry.content in t3_reply, (
            f"Case 3 Failure: Stub response '{t3_reply}' did not inject retrieved content '{top_memory.entry.content}'"
        )
        print(f"Assistant Reply: {t3_reply!r}")
        print("[OK] CASE 3 PASSED: Retrieved memory content successfully injected into assistant response.")

        # -------------------------------------------------------------------------
        # CASE 2: The Cross-Session Exclusion Proof
        # -------------------------------------------------------------------------
        print("\n[CASE 2] Running cross-session exclusion proof...")

        # Turn 2 in Session 3: Ask a follow-up turn
        s3_followup_query = "What kind of non-seafood restaurants do you recommend?"
        s3_reply2, retrieved_s3_turn2 = chatbot.process_user_turn(
            session_id=session_3,
            user_message=s3_followup_query,
        )

        retrieved_sessions = [m.entry.session_id for m in retrieved_s3_turn2]
        print(f"Session 3 Turn 2 Query: {s3_followup_query!r}")
        print(f"Retrieved memory session IDs: {retrieved_sessions}")

        for mem in retrieved_s3_turn2:
            assert mem.entry.session_id != session_3, (
                f"Case 2 Failure: Current session entry '{session_3}' leaked into retrieval results!"
            )
        print("[OK] CASE 2 PASSED: Interactions from current session (session_3) strictly excluded from retrieval.")

        # -------------------------------------------------------------------------
        # CASE 4: The Scoring Sanity Check
        # -------------------------------------------------------------------------
        print("\n[CASE 4] Running recency vs relevance scoring sanity check...")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir_c4:

            temp_path_c4 = Path(temp_dir_c4)
            db_c4 = temp_path_c4 / "c4_episodic.db"
            chroma_c4 = temp_path_c4 / "c4_chroma"

            mem_sys_c4 = MemorySystem(db_path=db_c4, chroma_path=chroma_c4)

            now_dt = datetime.now(timezone.utc)
            # Memory A: 1 hour old
            ts_recent = (now_dt - timedelta(hours=1)).isoformat()
            mem_sys_c4.add_entry(
                session_id="session_recent",
                role="user",
                content="I love eating red apples for breakfast.",
                timestamp=ts_recent,
            )

            # Memory B: 100 hours old
            ts_older = (now_dt - timedelta(hours=100)).isoformat()
            mem_sys_c4.add_entry(
                session_id="session_older",
                role="user",
                content="I love eating red apples for breakfast.",
                timestamp=ts_older,
            )

            # Query for apples from a new session
            query_c4 = "What fruit do I like to eat for breakfast?"
            retrieved_c4 = mem_sys_c4.retrieve_memories(
                query=query_c4,
                current_session_id="session_current",
                top_k=5,
                now=now_dt,
            )

            print("Scoring Sanity Check Results:")
            for m in retrieved_c4:
                print(
                    f"  Session: {m.entry.session_id} | "
                    f"Recency Score: {m.recency_score:.4f} | "
                    f"Relevance Score: {m.relevance_score:.4f} | "
                    f"Composite Score: {m.composite_score:.4f}"
                )

            assert len(retrieved_c4) == 2, "Expected 2 candidate memories"
            recent_mem = next(m for m in retrieved_c4 if m.entry.session_id == "session_recent")
            older_mem = next(m for m in retrieved_c4 if m.entry.session_id == "session_older")

            assert recent_mem.composite_score > older_mem.composite_score, (
                f"Case 4 Failure: Recent memory score ({recent_mem.composite_score}) was not strictly greater "
                f"than older memory score ({older_mem.composite_score})"
            )

            assert retrieved_c4[0].entry.session_id == "session_recent", (
                f"Case 4 Failure: Top ranked memory was '{retrieved_c4[0].entry.session_id}', expected 'session_recent'"
            )
            print(
                f"[OK] CASE 4 PASSED: Recent memory ('session_recent', composite={recent_mem.composite_score:.4f}) "

                f"scored higher than older memory ('session_older', composite={older_mem.composite_score:.4f})."
            )

    print("\n" + "=" * 70)
    print("ALL 4 SMOKE TEST PROOF CASES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_tests()
