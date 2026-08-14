"""
YAML to LangGraph StateGraph dynamic compiler.
"""

from __future__ import annotations

import os
import sys

# Ensure project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

import operator
from typing import Annotated, Any, Callable, Dict, Optional, TypedDict
# pyrefly: ignore [missing-import]
import yaml

# pyrefly: ignore [missing-import]
from langgraph.graph import END, START, StateGraph
# pyrefly: ignore [missing-import]
from langgraph.types import interrupt

# pyrefly: ignore [missing-import]
from src.registry import FUNCTION_REGISTRY
# pyrefly: ignore [missing-import]
from src.schema import WorkflowSpec


TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def build_dynamic_state_type(spec: WorkflowSpec) -> type:
    """Dynamically creates a TypedDict class based on the YAML workflow state specification."""
    fields: Dict[str, Any] = {}
    for field_spec in spec.state:
        base_type = TYPE_MAP.get(field_spec.type, str)
        if field_spec.reducer == "append":
            fields[field_spec.field] = Annotated[list, operator.add]
        else:
            fields[field_spec.field] = base_type
    return TypedDict("DynamicWorkflowState", fields)


def make_llm_node(node_spec, llm_factory: Optional[Callable] = None):
    """Creates a graph node callable for an llm_call step."""
    def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = node_spec.prompt_template.format(**state)
        
        if llm_factory is not None:
            llm = llm_factory(node_spec)
            res = llm.invoke(prompt)
            output_text = res.content if hasattr(res, "content") else str(res)
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY environment variable is missing. Cannot execute LLM node.")
            try:
                # pyrefly: ignore [missing-import]
                from langchain_groq import ChatGroq
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=node_spec.temperature or 0.0)
                res = llm.invoke(prompt)
                output_text = res.content if hasattr(res, "content") else str(res)
            except Exception as e:
                raise RuntimeError(f"Groq LLM invocation failed: {e}") from e

        return {node_spec.output_field: output_text}

    return node_fn


def make_function_node(node_spec):
    """Creates a graph node callable for a function step."""
    fn_name = node_spec.function_name
    if fn_name not in FUNCTION_REGISTRY:
        raise ValueError(f"Function '{fn_name}' requested by node '{node_spec.id}' is not in FUNCTION_REGISTRY.")
    fn = FUNCTION_REGISTRY[fn_name]

    def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        return fn(state)

    return node_fn


def make_human_review_node(node_spec):
    """Creates a graph node callable for a human_review interrupt step."""
    payload_fields = node_spec.payload_fields or []
    resume_field = node_spec.resume_field

    def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        payload = {field: state.get(field) for field in payload_fields}
        payload["instructions"] = f"Provide resume value for field '{resume_field}'."
        decision = interrupt(payload)
        return {resume_field: decision}

    return node_fn


def compile_workflow_from_yaml(
    yaml_content: str,
    checkpointer=None,
    llm_factory: Optional[Callable] = None,
):
    """Parses raw YAML, validates schema, compiles into a runnable LangGraph graph."""
    raw_data = yaml.safe_load(yaml_content)
    spec = WorkflowSpec.model_validate(raw_data)
    return compile_workflow(spec, checkpointer=checkpointer, llm_factory=llm_factory), spec


def compile_workflow(
    spec: WorkflowSpec,
    checkpointer=None,
    llm_factory: Optional[Callable] = None,
):
    """Compiles a WorkflowSpec model into a LangGraph StateGraph instance."""
    StateClass = build_dynamic_state_type(spec)
    builder = StateGraph(StateClass)

    # 1. Add Nodes
    for node_spec in spec.nodes:
        if node_spec.type == "llm_call":
            builder.add_node(node_spec.id, make_llm_node(node_spec, llm_factory=llm_factory))
        elif node_spec.type == "function":
            builder.add_node(node_spec.id, make_function_node(node_spec))
        elif node_spec.type == "human_review":
            builder.add_node(node_spec.id, make_human_review_node(node_spec))

    # 2. Add Fixed Edges
    for edge in spec.edges:
        src = START if edge.from_node == "START" else edge.from_node
        dst = END if edge.to_node == "END" else edge.to_node
        builder.add_edge(src, dst)

    # 3. Add Conditional Edges
    for c_edge in spec.conditional_edges:
        field_name = c_edge.field
        routes = c_edge.routes
        default_target = c_edge.default

        def make_router(f_name, r_table, def_t):
            def router(state: Dict[str, Any]) -> str:
                val = str(state.get(f_name, ""))
                target = r_table.get(val, def_t)
                return END if target == "END" else target
            return router

        # Build explicit path map for LangGraph
        path_map = {}
        for k, v in routes.items():
            target_node = END if v == "END" else v
            path_map[k] = target_node
            path_map[v] = target_node

        def_node = END if default_target == "END" else default_target
        path_map[default_target] = def_node
        path_map[END] = END
        path_map["END"] = END

        builder.add_conditional_edges(
            c_edge.from_node,
            make_router(field_name, routes, default_target),
            path_map,
        )

    return builder.compile(checkpointer=checkpointer)
