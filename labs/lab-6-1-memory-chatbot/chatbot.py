"""
chatbot.py — Memory-augmented chatbot supporting stub and real Groq LLM response modes.

Architecture:
- On user turn: logs user message, retrieves top cross-session memories, generates reply, logs assistant reply.
- Stub mode (default): deterministic template weaving top retrieved memory's content into response.
- Real mode (--real): ChatGroq(model="llama-3.3-70b-versatile", temperature=0) with memory injection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

try:
    # pyrefly: ignore [missing-import]
    from .memory_system import MemorySystem
    # pyrefly: ignore [missing-import]
    from .models import RetrievedMemory
except ImportError:
    # pyrefly: ignore [missing-import]
    from memory_system import MemorySystem
    # pyrefly: ignore [missing-import]
    from models import RetrievedMemory



class MemoryChatbot:
    """
    Memory-augmented chatbot that uses MemorySystem for episodic logging and cross-session retrieval.
    """

    def __init__(
        self,
        memory_system: MemorySystem,
        use_real: bool = False,
    ) -> None:
        self.memory_system = memory_system
        self.use_real = use_real

    def process_user_turn(
        self,
        session_id: str,
        user_message: str,
        timestamp: str | None = None,
    ) -> Tuple[str, List[RetrievedMemory]]:
        """
        Process a single user interaction turn end-to-end:
        1. Log user message into episodic and semantic stores.
        2. Retrieve top 5 cross-session memories (excluding current session_id).
        3. Generate assistant response (stub or real Groq LLM).
        4. Log assistant reply into episodic and semantic stores.
        5. Return (reply_text, retrieved_memories).
        """
        # 1. Store user turn
        self.memory_system.add_entry(
            session_id=session_id,
            role="user",
            content=user_message,
            timestamp=timestamp,
        )

        # 2. Retrieve top cross-session memories
        retrieved_memories = self.memory_system.retrieve_memories(
            query=user_message,
            current_session_id=session_id,
            top_k=5,
        )

        # 3. Generate response
        if self.use_real:
            reply_text = self._generate_real_reply(user_message, retrieved_memories)
        else:
            reply_text = self._generate_stub_reply(user_message, retrieved_memories)

        # 4. Store assistant reply turn
        self.memory_system.add_entry(
            session_id=session_id,
            role="assistant",
            content=reply_text,
            timestamp=timestamp,
        )

        return reply_text, retrieved_memories

    def _generate_stub_reply(
        self,
        user_message: str,
        memories: List[RetrievedMemory],
    ) -> str:
        """
        Deterministic stub response generator that injects top memory content.
        """
        if memories:
            top_mem = memories[0].entry
            return (
                f"Based on what you mentioned before ({top_mem.content!r}), "
                f"I can help answer your question: {user_message}"
            )
        return f"I've noted that: {user_message!r}. How else can I assist you?"

    def _generate_real_reply(
        self,
        user_message: str,
        memories: List[RetrievedMemory],
    ) -> str:
        """
        Real LLM reply generator using ChatGroq. Raises RuntimeError if GROQ_API_KEY is missing.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is required for --real mode."
            )

        # pyrefly: ignore [missing-import]
        from langchain_groq import ChatGroq

        context_lines = []
        if memories:
            for idx, mem in enumerate(memories, start=1):
                context_lines.append(
                    f"[{idx}] Session '{mem.entry.session_id}' ({mem.entry.timestamp}): {mem.entry.content}"
                )
            context_str = "\n".join(context_lines)
        else:
            context_str = "No prior relevant session memories found."

        prompt = (
            "You are a helpful AI assistant with access to long-term memory across user sessions.\n"
            "Below are relevant past memories retrieved from previous sessions:\n\n"
            f"{context_str}\n\n"
            "Answer the user's message accurately utilizing past session memory when relevant.\n"
            f"User message: {user_message}"
        )

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(prompt)
        return str(response.content).strip()


def make_stub_chatbot(
    db_path: str | Path | None = None,
    chroma_path: str | Path | None = None,
) -> MemoryChatbot:
    """Factory creating an offline deterministic stub chatbot."""
    mem_sys = MemorySystem(db_path=db_path, chroma_path=chroma_path)
    return MemoryChatbot(memory_system=mem_sys, use_real=False)


def make_real_chatbot(
    db_path: str | Path | None = None,
    chroma_path: str | Path | None = None,
) -> MemoryChatbot:
    """Factory creating a real Groq-powered chatbot."""
    mem_sys = MemorySystem(db_path=db_path, chroma_path=chroma_path)
    return MemoryChatbot(memory_system=mem_sys, use_real=True)
