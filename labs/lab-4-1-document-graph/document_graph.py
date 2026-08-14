"""
Lab 4.1 -- Document Processing Graph

load -> validate -> [conditional: invalid / oversized / normal]
                        invalid  -> END
                        oversized -> split -> chunk -> embed -> confirm
                        normal              -> chunk -> embed -> confirm

The point of this lab isn't the document processing itself (that part is
the same RecursiveCharacterTextSplitter + embeddings work from Weeks 1
and 3). The point is the conditional edge: `validate` inspects the state
at runtime and decides which node runs next -- something a linear LCEL
chain can't express, since a chain's shape is fixed the moment you write
`a | b | c`. A graph's shape can depend on the data.

Usage:
    python document_graph.py process sample_docs/short_note.txt
    python document_graph.py process sample_docs/long_report.txt
    python document_graph.py process sample_docs/does_not_exist.txt
"""

from pathlib import Path
from typing import TypedDict

# pyrefly: ignore [missing-import]
import typer
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = typer.Typer()
console = Console()

# A doc past this many characters gets coarse-split into parts before
# fine-grained chunking. In a real repo this should be tuned to whatever
# "oversized" means for your actual documents (often token-based, not
# character-based) -- kept small here so the two branches are easy to
# demo with short sample files.
OVERSIZED_THRESHOLD_CHARS = 4000

# Coarse split, only used on oversized docs, before the normal chunker
# runs on each part.
PART_SIZE, PART_OVERLAP = 2000, 100

# Same chunk size used for RAG in Week 1 / Week 3, for consistency.
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class DocumentState(TypedDict, total=False):
    file_path: str
    raw_text: str
    char_count: int
    is_valid: bool
    validation_message: str
    is_oversized: bool
    parts: list[str]
    chunks: list[str]
    embedding_dim: int
    num_embedded: int
    confirmed: bool
    summary: str


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def load_node(state: DocumentState) -> dict:
    """Read the file straight off disk. Plain Python, no loader
    abstraction needed for a single text file -- langchain_community's
    TextLoader would pull in a dependency that's now being sunset in
    favour of standalone integration packages, for something this simple
    it isn't worth it."""
    path = Path(state["file_path"])
    if not path.exists():
        return {"raw_text": "", "char_count": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"raw_text": text, "char_count": len(text)}


def validate_node(state: DocumentState) -> dict:
    """Decide validity AND oversized-ness in one place, since both feed
    the same conditional edge right after this node."""
    path = Path(state["file_path"])

    if not path.exists():
        return {
            "is_valid": False,
            "validation_message": f"File not found: {state['file_path']}",
            "is_oversized": False,
        }

    if not state["raw_text"].strip():
        return {
            "is_valid": False,
            "validation_message": "File exists but is empty (or whitespace only).",
            "is_oversized": False,
        }

    is_oversized = state["char_count"] > OVERSIZED_THRESHOLD_CHARS
    return {
        "is_valid": True,
        "validation_message": "OK",
        "is_oversized": is_oversized,
    }


def route_after_validate(state: DocumentState) -> str:
    """The routing function a conditional edge calls. It only reads
    state and returns a string key -- LangGraph maps that key to a
    target node via the dict passed to add_conditional_edges."""
    if not state["is_valid"]:
        return "invalid"
    if state["is_oversized"]:
        return "oversized"
    return "normal"


def split_node(state: DocumentState) -> dict:
    """Coarse split for oversized docs only. Reached exclusively via the
    'oversized' branch -- normal-sized docs skip this node entirely."""
    coarse_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PART_SIZE, chunk_overlap=PART_OVERLAP
    )
    parts = coarse_splitter.split_text(state["raw_text"])
    return {"parts": parts}


def chunk_node(state: DocumentState) -> dict:
    """Reached from both branches, so it has to handle either case:
    'parts' populated (came via split_node) or absent (came directly
    # pyrefly: ignore [missing-import]
    from validate_node)."""
    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    parts = state.get("parts") or [state["raw_text"]]
    chunks: list[str] = []
    for part in parts:
        chunks.extend(fine_splitter.split_text(part))

    return {"chunks": chunks}


def make_embed_node(embedding_model):
    """Closure so the embedding model is injected rather than hardcoded
    -- build_graph(embedding_model=FakeEmbeddings(...)) is how this gets
    smoke-tested without downloading a real model."""

    def embed_node(state: DocumentState) -> dict:
        vectors = embedding_model.embed_documents(state["chunks"])
        return {
            "embedding_dim": len(vectors[0]) if vectors else 0,
            "num_embedded": len(vectors),
        }

    return embed_node


def confirm_node(state: DocumentState) -> dict:
    if state.get("is_oversized"):
        summary = (
            f"{state['file_path']}: {state['char_count']} chars -> "
            f"split into {len(state.get('parts', []))} parts -> "
            f"{len(state['chunks'])} chunks -> "
            f"{state['num_embedded']} embeddings (dim={state['embedding_dim']})"
        )
    else:
        summary = (
            f"{state['file_path']}: {state['char_count']} chars -> "
            f"{len(state['chunks'])} chunks -> "
            f"{state['num_embedded']} embeddings (dim={state['embedding_dim']})"
        )
    return {"confirmed": True, "summary": summary}


# ---------------------------------------------------------------------------
# GRAPH ASSEMBLY
# ---------------------------------------------------------------------------
def get_embedding_model():
    """Real embedding model, same one used since Week 1. Imported lazily
    so this module doesn't require sentence-transformers/torch just to
    be imported for testing with a fake embedder."""
    # pyrefly: ignore [missing-import]
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_graph(embedding_model=None):
    if embedding_model is None:
        embedding_model = get_embedding_model()

    builder = StateGraph(DocumentState)
    builder.add_node("load", load_node)
    builder.add_node("validate", validate_node)
    builder.add_node("split", split_node)
    builder.add_node("chunk", chunk_node)
    builder.add_node("embed", make_embed_node(embedding_model))
    builder.add_node("confirm", confirm_node)

    builder.add_edge(START, "load")
    builder.add_edge("load", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"invalid": END, "oversized": "split", "normal": "chunk"},
    )
    builder.add_edge("split", "chunk")
    builder.add_edge("chunk", "embed")
    builder.add_edge("embed", "confirm")
    builder.add_edge("confirm", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def process(file_path: str):
    """Run a document through the graph and print the result."""
    graph = build_graph()
    result = graph.invoke({"file_path": file_path, "parts": []})

    if not result.get("is_valid", False):
        console.print(
            Panel(
                result.get("validation_message", "Unknown validation error"),
                title="Rejected at validation",
                style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(Panel(result["summary"], title="Processed", style="green"))


if __name__ == "__main__":
    app()
