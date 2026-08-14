"""
Lab 4.3 -- Human-in-the-Loop Content Moderation Graph
CalderR Agentic AI Engineering Internship, Week 4

Posts are classified as approve / reject / borderline. Approve and reject
resolve automatically. Borderline pauses the graph via interrupt() and
waits for a human decision. The pause is durable across a process
restart because the graph is compiled with SqliteSaver (a real file on
disk), not InMemorySaver/MemorySaver, which would lose the pending
review the instant this process exited.

Usage:
    python moderation_graph.py submit "post content" [--real]
    python moderation_graph.py resume <thread_id> <approve|reject>
    python moderation_graph.py status <thread_id>

--real switches the classifier from the built-in offline stub to an
actual Groq call (llama-3.3-70b-versatile via langchain-groq), matching
the rest of the repo's stack. Requires GROQ_API_KEY in the environment.
Default is the stub, so this file runs end-to-end with zero network
access and zero credentials required.
"""

from __future__ import annotations

import argparse
import operator
import os
import sys
import uuid
from typing import Annotated, Callable, Literal, Optional, TypedDict

# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite import SqliteSaver
# pyrefly: ignore [missing-import]
from langgraph.graph import END, START, StateGraph
# pyrefly: ignore [missing-import]
from langgraph.types import Command, interrupt
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moderation_state.db")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ModerationState(TypedDict):
    post_id: str
    content: str
    classification: Literal["approve", "reject", "borderline", ""]
    reasoning: str
    human_decision: Optional[Literal["approve", "reject"]]
    final_decision: Optional[Literal["approve", "reject"]]
    # Annotated + operator.add makes this an append-only reducer instead of
    # last-write-wins: every node contributes to one growing audit trail
    # instead of each node's return value overwriting the last node's.
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Classifiers -- injected into the graph rather than hardcoded, so the
# graph's mechanics (routing / interrupt / persistence) can be exercised
# without network access or an API key. See build_classify_node below.
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    label: Literal["approve", "reject", "borderline"]
    reasoning: str = Field(description="One sentence explaining the call")


def make_real_classifier() -> Callable[[str], tuple[str, str]]:
    """Groq-backed classifier -- matches the rest of the repo's stack
    (llama-3.3-70b-versatile via ChatGroq). Requires GROQ_API_KEY."""
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it, or drop --real to use "
            "the offline stub classifier instead."
        )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(
        ClassificationResult
    )

    def classify(content: str) -> tuple[str, str]:
        result = llm.invoke(
            "You are a content moderation classifier. Classify the post as "
            "'approve' (clearly fine), 'reject' (clearly violates policy: "
            "harassment, spam, illegal content), or 'borderline' (unclear, "
            "needs a human). Be conservative: prefer 'borderline' over a "
            f"confident guess either way.\n\nPost:\n{content}"
        )
        return result.label, result.reasoning

    return classify


# NOT a real moderation policy -- just enough logic to exercise all three
# routing branches (approve / reject / borderline) with zero network calls,
# so the graph's mechanics can be verified offline and deterministically.
_BANNED_TERMS = {"scamlink", "freemoneyclick", "hackpassword"}


def make_stub_classifier() -> Callable[[str], tuple[str, str]]:
    def classify(content: str) -> tuple[str, str]:
        lowered = content.lower()
        if any(term in lowered for term in _BANNED_TERMS):
            return "reject", "Matched a blocked-term entry."
        if "urgent" in lowered or "??" in content or "scam" in lowered:
            return "borderline", "Urgency/ambiguity markers present -- needs human judgment."
        return "approve", "No policy signals detected."

    return classify


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def build_classify_node(classifier: Callable[[str], tuple[str, str]]):
    def classify_post(state: ModerationState) -> dict:
        label, reasoning = classifier(state["content"])
        return {
            "classification": label,
            "reasoning": reasoning,
            "log": [f"classified '{state['post_id']}' as {label}: {reasoning}"],
        }

    return classify_post


def human_review_node(state: ModerationState) -> dict:
    # interrupt() has to be the first thing this node can do. On resume,
    # LangGraph re-executes this whole function from the top rather than
    # continuing mid-line -- everything above the interrupt() call reruns
    # too. Here that's just two dict lookups building the payload, which
    # is harmless to repeat. If this node wrote a log line or fired a
    # notification *before* interrupt(), that side effect would fire
    # again on every single resume.
    decision = interrupt(
        {
            "post_id": state["post_id"],
            "content": state["content"],
            "auto_classification": state["classification"],
            "auto_reasoning": state["reasoning"],
            "instructions": "Resume with 'approve' or 'reject'.",
        }
    )
    return {
        "human_decision": decision,
        "log": [f"human review on '{state['post_id']}': {decision}"],
    }


def finalize_decision(state: ModerationState) -> dict:
    # decided_by is derived from what's actually present in state (was a
    # human decision recorded?) rather than tracked as its own separate
    # flag -- one fewer place for the two to drift out of sync.
    if state.get("human_decision") is not None:
        final = state["human_decision"]
        decided_by = "human"
    else:
        final = state["classification"]
        decided_by = "auto"
    return {
        "final_decision": final,
        "log": [f"finalized '{state['post_id']}': {final} (via {decided_by})"],
    }


def route_after_classification(state: ModerationState) -> str:
    return "human_review" if state["classification"] == "borderline" else "finalize"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph(classifier: Callable[[str], tuple[str, str]], checkpointer):
    builder = StateGraph(ModerationState)
    builder.add_node("classify", build_classify_node(classifier))
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_decision)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classification,
        {"human_review": "human_review", "finalize": "finalize"},
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_submit(args):
    classifier = make_real_classifier() if args.real else make_stub_classifier()
    thread_id = str(uuid.uuid4())
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(classifier, checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        initial: ModerationState = {
            "post_id": thread_id[:8],
            "content": args.content,
            "classification": "",
            "reasoning": "",
            "human_decision": None,
            "final_decision": None,
            "log": [],
        }
        result = graph.invoke(initial, config=config)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"PENDING HUMAN REVIEW  thread_id={thread_id}")
            print(f"  auto-classification: {payload['auto_classification']} ({payload['auto_reasoning']})")
            print(f"  content: {payload['content']}")
            print(f"  -> resume with: python moderation_graph.py resume {thread_id} <approve|reject>")
        else:
            print(f"AUTO-DECIDED  thread_id={thread_id}  final_decision={result['final_decision']}")
            for line in result["log"]:
                print(f"  - {line}")


def cmd_resume(args):
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        # The classifier is never actually called on resume -- the graph
        # is already past "classify" per the checkpoint -- so the stub is
        # fine here regardless of how the post was originally submitted.
        graph = build_graph(make_stub_classifier(), checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

        snapshot = graph.get_state(config)
        if not snapshot.next:
            print(
                f"No pending review for thread_id={args.thread_id} "
                f"(already finalized, or thread_id doesn't exist)."
            )
            sys.exit(1)

        result = graph.invoke(Command(resume=args.decision), config=config)

    print(f"RESOLVED  thread_id={args.thread_id}  final_decision={result['final_decision']}")
    for line in result["log"]:
        print(f"  - {line}")


def cmd_status(args):
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(make_stub_classifier(), checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}
        snapshot = graph.get_state(config)

    if not snapshot.values:
        print(f"No such thread_id: {args.thread_id}")
        sys.exit(1)

    pending = bool(snapshot.next)
    print(f"thread_id={args.thread_id}  pending_human_review={pending}")
    print(f"  next node(s): {snapshot.next or '(none -- finalized)'}")
    for line in snapshot.values.get("log", []):
        print(f"  - {line}")


def main():
    parser = argparse.ArgumentParser(description="Lab 4.3 -- HITL content moderation graph")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit a post for moderation")
    p_submit.add_argument("content")
    p_submit.add_argument("--real", action="store_true", help="Use the Groq classifier instead of the offline stub")
    p_submit.set_defaults(func=cmd_submit)

    p_resume = sub.add_parser("resume", help="Resume a pending human review")
    p_resume.add_argument("thread_id")
    p_resume.add_argument("decision", choices=["approve", "reject"])
    p_resume.set_defaults(func=cmd_resume)

    p_status = sub.add_parser("status", help="Check whether a thread is pending or finalized")
    p_status.add_argument("thread_id")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        # Expected, user-fixable failures (missing GROQ_API_KEY, etc.) --
        # print the message and exit cleanly instead of a raw traceback.
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
