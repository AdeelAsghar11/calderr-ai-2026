"""
Tuesday core learning -- classification router.

query -> classify -> [conditional: general / technical / sensitive] -> handler -> END

The router in Lab 4.1 branched on a mechanical fact (character count). This
one branches on an LLM's judgment call, which is the more common real-world
shape: classify -> route -> specialized system prompt per branch.

Usage:
    python 00_classification_router.py route "My Docker container keeps crashing"
"""

from typing import Literal, TypedDict

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END

load_dotenv()

app = typer.Typer()
console = Console()

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user's query into exactly one category.\n"
    "- general: everyday questions, small talk, casual topics\n"
    "- technical: programming, software, engineering, debugging questions\n"
    "- sensitive: legal, medical, financial, HR, or other topics where the "
    "person needs professional advice, not a casual answer"
)

HANDLER_PROMPTS = {
    "general": "You are a friendly, helpful general-purpose assistant.",
    "technical": (
        "You are a precise technical assistant. Give accurate, specific, "
        "actionable answers."
    ),
    "sensitive": (
        "You are a careful assistant handling a sensitive query. Give only "
        "general, factual information, and clearly recommend the person "
        "consult an appropriate professional (lawyer, doctor, financial "
        "advisor, or HR) for advice specific to their situation."
    ),
}


class QueryClassification(BaseModel):
    category: Literal["general", "technical", "sensitive"]
    reasoning: str = Field(description="one short sentence explaining the classification")


class RouterState(TypedDict, total=False):
    query: str
    category: str
    response: str


# ---------------------------------------------------------------------------
# Real backends (Groq). Imported lazily so this module doesn't require an
# API key just to be imported for testing with fakes.
# ---------------------------------------------------------------------------
def default_classify(query: str) -> str:
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    classifier = llm.with_structured_output(QueryClassification)
    result = classifier.invoke(f"{CLASSIFY_SYSTEM_PROMPT}\n\nQuery: {query}")
    return result.category


def default_respond(query: str, category: str) -> str:
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq
    # pyrefly: ignore [missing-import]
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    messages = [
        SystemMessage(content=HANDLER_PROMPTS[category]),
        HumanMessage(content=query),
    ]
    return llm.invoke(messages).content


# ---------------------------------------------------------------------------
# Nodes -- classify_fn / respond_fn are injected so build_graph() can be
# smoke-tested without hitting Groq. See smoke_test.py.
# ---------------------------------------------------------------------------
def make_classify_node(classify_fn):
    def classify_node(state: RouterState) -> dict:
        return {"category": classify_fn(state["query"])}

    return classify_node


def make_handler_node(respond_fn, category: str):
    def handler_node(state: RouterState) -> dict:
        return {"response": respond_fn(state["query"], category)}

    return handler_node


def route_by_category(state: RouterState) -> str:
    # The classifier's category and the node names are identical strings
    # here, so the mapping below is just an identity lookup. Worth keeping
    # explicit anyway: it decouples the routing function from node names,
    # same as Monday's non-identity mapping ("oversized" -> "split").
    return state["category"]


def build_graph(classify_fn=None, respond_fn=None):
    classify_fn = classify_fn or default_classify
    respond_fn = respond_fn or default_respond

    builder = StateGraph(RouterState)
    builder.add_node("classify", make_classify_node(classify_fn))
    builder.add_node("general", make_handler_node(respond_fn, "general"))
    builder.add_node("technical", make_handler_node(respond_fn, "technical"))
    builder.add_node("sensitive", make_handler_node(respond_fn, "sensitive"))

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {"general": "general", "technical": "technical", "sensitive": "sensitive"},
    )
    builder.add_edge("general", END)
    builder.add_edge("technical", END)
    builder.add_edge("sensitive", END)

    return builder.compile()


@app.command()
def route(query: str):
    graph = build_graph()
    result = graph.invoke({"query": query})
    console.print(Panel(result["response"], title=f"category: {result['category']}"))


if __name__ == "__main__":
    app()
