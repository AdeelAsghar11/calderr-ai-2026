"""
Lab 4.2 -- Self-Correcting Agent Loop

generate -> validate -> [conditional: pass / retry / give_up]
                            pass    -> respond -> END
                            retry   -> generate   (cycle)
                            give_up -> give_up -> END

Task: write an 8-word-or-fewer tagline that includes the product name.
Validation is plain Python (word count, substring check), deliberately not
another LLM call -- cheap, deterministic, and the point of this lab is the
loop mechanics, not judge quality. Swap in an LLM-as-judge (RAGAS-style,
like Week 3) for criteria that genuinely can't be checked by rule.

Usage:
    python self_correcting_loop.py generate "Brew" "small-batch coffee delivered weekly"
"""

import logging
from typing import Optional, TypedDict

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from langgraph.graph import StateGraph, START, END

load_dotenv()

app = typer.Typer()
console = Console()

MAX_ATTEMPTS = 3
MAX_WORDS = 8

logger = logging.getLogger("self_correcting_loop")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # logging.basicConfig() is a no-op if the root logger already has a
    # handler attached (langgraph or another import can trigger that
    # first), so this attaches a handler directly to a named logger
    # instead of relying on basicConfig. propagate=False stops messages
    # from also hitting any root handler and printing twice.
    _handler = logging.FileHandler("loop_log.txt", mode="a")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


class TaglineState(TypedDict, total=False):
    product_name: str
    brief: str
    max_attempts: int
    attempt: int
    draft: str
    is_valid: bool
    validation_feedback: str
    final_response: str
    gave_up: bool
    iterations_used: int


# ---------------------------------------------------------------------------
# Real backend (Groq). Imported lazily, not needed for the smoke test.
# ---------------------------------------------------------------------------
def default_generate(product_name: str, brief: str, feedback: Optional[str] = None) -> str:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    prompt = (
        f"Write a marketing tagline for a product called '{product_name}': {brief}. "
        f"Maximum {MAX_WORDS} words. Must include the product name '{product_name}'."
    )
    if feedback:
        prompt += f" Your previous attempt had this problem: {feedback}. Fix it."
    return llm.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# Nodes -- generate_fn is injected. See smoke_test.py for the scripted
# fake used to prove the loop actually retries and terminates correctly.
# ---------------------------------------------------------------------------
def make_generate_node(generate_fn):
    def generate_node(state: TaglineState) -> dict:
        attempt = state.get("attempt", 0) + 1
        feedback = state.get("validation_feedback") or None
        draft = generate_fn(state["product_name"], state["brief"], feedback)
        logger.info(
            "attempt=%d product=%s draft=%r feedback_used=%r",
            attempt, state["product_name"], draft, feedback,
        )
        return {"draft": draft, "attempt": attempt}

    return generate_node


def validate_node(state: TaglineState) -> dict:
    draft = state["draft"]
    word_count = len(draft.split())
    issues = []
    if word_count > MAX_WORDS:
        issues.append(f"{word_count} words, needs to be {MAX_WORDS} or fewer")
    if state["product_name"].lower() not in draft.lower():
        issues.append(f"missing the product name '{state['product_name']}'")

    is_valid = not issues
    feedback = "; ".join(issues)
    logger.info(
        "attempt=%d product=%s valid=%s feedback=%r",
        state["attempt"], state["product_name"], is_valid, feedback,
    )
    return {"is_valid": is_valid, "validation_feedback": feedback}


def route_after_validate(state: TaglineState) -> str:
    if state["is_valid"]:
        return "pass"
    if state["attempt"] >= state.get("max_attempts", MAX_ATTEMPTS):
        return "give_up"
    return "retry"


def respond_node(state: TaglineState) -> dict:
    logger.info(
        "DONE product=%s result=passed iterations=%d",
        state["product_name"], state["attempt"],
    )
    return {
        "final_response": state["draft"],
        "gave_up": False,
        "iterations_used": state["attempt"],
    }


def give_up_node(state: TaglineState) -> dict:
    logger.info(
        "DONE product=%s result=gave_up iterations=%d last_issue=%r",
        state["product_name"], state["attempt"], state["validation_feedback"],
    )
    return {
        "final_response": state["draft"],
        "gave_up": True,
        "iterations_used": state["attempt"],
    }


def build_graph(generate_fn=None):
    generate_fn = generate_fn or default_generate

    builder = StateGraph(TaglineState)
    builder.add_node("generate", make_generate_node(generate_fn))
    builder.add_node("validate", validate_node)
    builder.add_node("respond", respond_node)
    builder.add_node("give_up", give_up_node)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"pass": "respond", "retry": "generate", "give_up": "give_up"},
    )
    builder.add_edge("respond", END)
    builder.add_edge("give_up", END)

    return builder.compile()


@app.command()
def generate(product_name: str, brief: str):
    graph = build_graph()
    result = graph.invoke(
        {"product_name": product_name, "brief": brief, "max_attempts": MAX_ATTEMPTS}
    )
    style = "red" if result["gave_up"] else "green"
    title = f"{'gave up after' if result['gave_up'] else 'passed after'} {result['iterations_used']} attempt(s)"
    console.print(Panel(result["final_response"], title=title, style=style))


if __name__ == "__main__":
    app()
