"""
Phase 3 Verification Tests
- All 5 example YAML workflows parse, validate schema, and compile without error
"""

import os
import sys
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.compiler import compile_workflow_from_yaml
from src.engine import WorkflowEngine

EXPECTED_WORKFLOW_FILES = [
    "1_function_pipeline.yaml",
    "2_human_approval.yaml",
    "3_linear_llm_pipeline.yaml",
    "4_llm_classification_branch.yaml",
    "5_cyclic_refinement.yaml",
]


def test_all_5_yaml_workflows_parse_and_compile():
    workflows_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workflows"))
    engine = WorkflowEngine()

    compiled_count = 0
    for filename in EXPECTED_WORKFLOW_FILES:
        filepath = os.path.join(workflows_dir, filename)
        assert os.path.exists(filepath), f"Missing workflow file: {filename}"

        spec = engine.register_yaml_file(filepath)
        assert spec.name is not None
        assert len(spec.nodes) > 0

        # Verify compilation into LangGraph graph instance
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        graph, compiled_spec = compile_workflow_from_yaml(content)
        assert graph is not None
        assert compiled_spec.name == spec.name
        compiled_count += 1

    assert compiled_count == 5
