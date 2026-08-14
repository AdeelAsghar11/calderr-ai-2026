"""
Schema definitions and Pydantic models for YAML Workflow validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, model_validator


class StateFieldSpec(BaseModel):
    field: str
    type: Literal["str", "int", "float", "bool", "list", "dict"]
    reducer: Literal["overwrite", "append"] = "overwrite"
    default: Optional[Any] = None


class NodeSpec(BaseModel):
    id: str
    type: Literal["llm_call", "function", "human_review"]
    # llm_call
    prompt_template: Optional[str] = None
    output_field: Optional[str] = None
    temperature: Optional[float] = 0.0
    # function
    function_name: Optional[str] = None
    # human_review
    payload_fields: Optional[List[str]] = None
    resume_field: Optional[str] = None

    @model_validator(mode="after")
    def validate_node_type_fields(self) -> NodeSpec:
        if self.type == "llm_call":
            if not self.prompt_template or not self.output_field:
                raise ValueError(
                    f"Node '{self.id}' of type 'llm_call' must specify 'prompt_template' and 'output_field'."
                )
        elif self.type == "function":
            if not self.function_name:
                raise ValueError(
                    f"Node '{self.id}' of type 'function' must specify 'function_name'."
                )
        elif self.type == "human_review":
            if not self.payload_fields or not self.resume_field:
                raise ValueError(
                    f"Node '{self.id}' of type 'human_review' must specify 'payload_fields' and 'resume_field'."
                )
        return self


class EdgeSpec(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")


class ConditionalEdgeSpec(BaseModel):
    from_node: str = Field(..., alias="from")
    field: str
    routes: Dict[str, str]
    default: str


class WorkflowSpec(BaseModel):
    name: str
    description: str
    state: List[StateFieldSpec]
    nodes: List[NodeSpec]
    edges: List[EdgeSpec] = Field(default_factory=list)
    conditional_edges: List[ConditionalEdgeSpec] = Field(default_factory=list)
    max_iterations: Optional[int] = None

    @model_validator(mode="after")
    def validate_graph_structure(self) -> WorkflowSpec:
        # Check node IDs uniqueness
        node_ids = set()
        for node in self.nodes:
            if node.id in node_ids:
                raise ValueError(f"Duplicate node id detected: '{node.id}'")
            node_ids.add(node.id)

        valid_targets = node_ids | {"END"}
        valid_sources = node_ids | {"START"}

        # Validate normal edges
        for edge in self.edges:
            if edge.from_node not in valid_sources:
                raise ValueError(
                    f"Edge source '{edge.from_node}' is not a valid node ID or 'START'."
                )
            if edge.to_node not in valid_targets:
                raise ValueError(
                    f"Edge target '{edge.to_node}' is not a valid node ID or 'END'."
                )

        # Validate conditional edges
        for c_edge in self.conditional_edges:
            if c_edge.from_node not in node_ids:
                raise ValueError(
                    f"Conditional edge source '{c_edge.from_node}' is not a valid node ID."
                )
            for val, target in c_edge.routes.items():
                if target not in valid_targets:
                    raise ValueError(
                        f"Conditional route target '{target}' for value '{val}' is not a valid node ID or 'END'."
                    )
            if c_edge.default not in valid_targets:
                raise ValueError(
                    f"Conditional default target '{c_edge.default}' is not a valid node ID or 'END'."
                )

        return self
