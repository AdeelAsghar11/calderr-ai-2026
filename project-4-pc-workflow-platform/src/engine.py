"""
Execution Engine backed by SqliteSaver for durable process state persistence.
"""

from __future__ import annotations

import os
import sys

# Ensure project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, Optional, Tuple
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

# pyrefly: ignore [missing-import]
from src.compiler import compile_workflow, compile_workflow_from_yaml
# pyrefly: ignore [missing-import]
from src.schema import WorkflowSpec


class WorkflowEngine:
    def __init__(self, db_path: str = "workflows_state.db"):
        self.db_path = db_path
        self._specs: Dict[str, WorkflowSpec] = {}
        self._yaml_sources: Dict[str, str] = {}

    def register_yaml(self, yaml_content: str) -> WorkflowSpec:
        """Parses, validates, and registers a YAML workflow definition."""
        import yaml
        raw_data = yaml.safe_load(yaml_content)
        spec = WorkflowSpec.model_validate(raw_data)
        self._specs[spec.name] = spec
        self._yaml_sources[spec.name] = yaml_content
        return spec

    def register_yaml_file(self, file_path: str) -> WorkflowSpec:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.register_yaml(content)

    def list_workflows(self) -> Dict[str, WorkflowSpec]:
        return dict(self._specs)

    def get_workflow_spec(self, name: str) -> Optional[WorkflowSpec]:
        return self._specs.get(name)

    def _prepare_initial_state(self, spec: WorkflowSpec, user_state: Dict[str, Any]) -> Dict[str, Any]:
        state = {}
        for f in spec.state:
            if f.field in user_state:
                state[f.field] = user_state[f.field]
            elif f.default is not None:
                state[f.field] = f.default
            else:
                if f.type == "str":
                    state[f.field] = ""
                elif f.type in ("int", "float"):
                    state[f.field] = 0
                elif f.type == "bool":
                    state[f.field] = False
                elif f.type == "list":
                    state[f.field] = []
                elif f.type == "dict":
                    state[f.field] = {}
        return state

    def run_workflow(
        self,
        workflow_name: str,
        initial_state: Dict[str, Any],
        thread_id: str,
        llm_factory: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Starts executing a workflow run from initial state."""
        spec = self._specs.get(workflow_name)
        if not spec:
            raise ValueError(f"Workflow '{workflow_name}' is not registered.")

        full_initial_state = self._prepare_initial_state(spec, initial_state)
        rec_limit = spec.max_iterations if spec.max_iterations is not None else 25

        with SqliteSaver.from_conn_string(self.db_path) as checkpointer:
            graph = compile_workflow(spec, checkpointer=checkpointer, llm_factory=llm_factory)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": rec_limit,
            }
            
            try:
                result = graph.invoke(full_initial_state, config=config)
            except GraphRecursionError as e:
                return {
                    "run_id": thread_id,
                    "status": "failed",
                    "error": f"max_iterations limit ({rec_limit}) exceeded",
                    "current_node": None,
                    "state": full_initial_state,
                }

            snapshot = graph.get_state(config)
            return self._format_status(thread_id, snapshot)

    def resume_workflow(
        self,
        workflow_name: str,
        thread_id: str,
        resume_value: Any,
        llm_factory: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Resumes a paused workflow run with user-provided resume value."""
        spec = self._specs.get(workflow_name)
        if not spec:
            raise ValueError(f"Workflow '{workflow_name}' is not registered.")

        rec_limit = spec.max_iterations if spec.max_iterations is not None else 25

        with SqliteSaver.from_conn_string(self.db_path) as checkpointer:
            graph = compile_workflow(spec, checkpointer=checkpointer, llm_factory=llm_factory)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": rec_limit,
            }
            
            snapshot = graph.get_state(config)
            if not snapshot.next:
                raise RuntimeError(f"Run '{thread_id}' is not in a paused state (already completed or invalid).")

            try:
                graph.invoke(Command(resume=resume_value), config=config)
            except GraphRecursionError:
                return {
                    "run_id": thread_id,
                    "status": "failed",
                    "error": f"max_iterations limit ({rec_limit}) exceeded",
                    "current_node": None,
                    "state": snapshot.values,
                }

            updated_snapshot = graph.get_state(config)
            return self._format_status(thread_id, updated_snapshot)

    def get_run_status(self, workflow_name: str, thread_id: str) -> Dict[str, Any]:
        """Retrieves the current state and execution status of a run from SQLite persistence."""
        spec = self._specs.get(workflow_name)
        if not spec:
            raise ValueError(f"Workflow '{workflow_name}' is not registered.")

        with SqliteSaver.from_conn_string(self.db_path) as checkpointer:
            graph = compile_workflow(spec, checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = graph.get_state(config)
            return self._format_status(thread_id, snapshot)

    def _format_status(self, thread_id: str, snapshot: Any) -> Dict[str, Any]:
        if not snapshot.values:
            return {
                "run_id": thread_id,
                "status": "not_found",
                "current_node": None,
                "state": {},
            }
        
        if snapshot.next:
            status = "paused"
            current_node = snapshot.next[0]
        else:
            status = "completed"
            current_node = None

        return {
            "run_id": thread_id,
            "status": status,
            "current_node": current_node,
            "state": snapshot.values,
        }
