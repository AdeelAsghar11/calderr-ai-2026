"""
Streamlit Dashboard for Workflow Orchestration Platform.
Provides workflow selection, Graphviz graph visualization, run controls,
live status view, and human-in-the-loop approval interface.
"""

import os
import sys
import json
import uuid
# pyrefly: ignore [missing-import]
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from src.engine import WorkflowEngine
# pyrefly: ignore [missing-import]
from src.schema import WorkflowSpec

DB_PATH = os.environ.get("WORKFLOW_DB_PATH", "workflows_dashboard_state.db")


@st.cache_resource
def get_engine():
    engine = WorkflowEngine(db_path=DB_PATH)
    workflows_dir = os.path.join(os.path.dirname(__file__), "..", "workflows")
    if os.path.exists(workflows_dir):
        for fname in os.listdir(workflows_dir):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(workflows_dir, fname)
                try:
                    engine.register_yaml_file(fpath)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
    return engine


def spec_to_graphviz(spec: WorkflowSpec) -> str:
    dot = [
        "digraph G {",
        '  rankdir="TB";',
        '  node [fontname="Helvetica", shape=box, style="filled,rounded", margin="0.2,0.1"];',
    ]
    for node in spec.nodes:
        if node.type == "llm_call":
            color = "#E3F2FD"  # soft blue
            border = "#1E88E5"
        elif node.type == "human_review":
            color = "#FFF3E0"  # soft orange
            border = "#FB8C00"
        else:
            color = "#E8F5E9"  # soft green
            border = "#43A047"
        dot.append(
            f'  "{node.id}" [label="{node.id}\\ntype: {node.type}", fillcolor="{color}", color="{border}"];'
        )

    dot.append('  "START" [shape=ellipse, fillcolor="#D1C4E9", color="#5E35B1"];')
    dot.append('  "END" [shape=ellipse, fillcolor="#FFCDD2", color="#E53935"];')

    for edge in spec.edges:
        dot.append(f'  "{edge.from_node}" -> "{edge.to_node}";')

    for c_edge in spec.conditional_edges:
        for val, target in c_edge.routes.items():
            dot.append(
                f'  "{c_edge.from_node}" -> "{target}" [label="{c_edge.field} == \'{val}\'", style=dashed, color="#757575"];'
            )
        if c_edge.default:
            dot.append(
                f'  "{c_edge.from_node}" -> "{c_edge.default}" [label="default", style=dotted, color="#9E9E9E"];'
            )

    dot.append("}")
    return "\n".join(dot)


def main():
    st.set_page_config(
        page_title="Workflow Orchestration Platform",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ Production Workflow Orchestration Platform")
    st.caption("Declarative YAML Workflow Engine with LangGraph Persistence & Status Streaming")

    engine = get_engine()
    workflows = engine.list_workflows()

    if not workflows:
        st.error("No YAML workflows found in `workflows/` directory.")
        return

    # Sidebar Selection
    st.sidebar.header("📋 Registered Workflows")
    selected_name = st.sidebar.selectbox(
        "Select Workflow",
        options=list(workflows.keys()),
        format_func=lambda k: f"{k} ({len(workflows[k].nodes)} nodes)",
    )

    spec = workflows[selected_name]

    # Tabs: Graph View | Run Workflow | Custom YAML Compiler
    tab_graph, tab_run, tab_compile = st.tabs(["📊 Graph Visualization", "🚀 Run Workflow", "📝 Compile New YAML"])

    with tab_graph:
        st.subheader(f"Workflow: `{spec.name}`")
        st.write(f"**Description**: {spec.description}")
        if spec.max_iterations:
            st.info(f"🔁 **Max Iterations Cap**: {spec.max_iterations}")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### Compiled Graph Diagram")
            dot_str = spec_to_graphviz(spec)
            st.graphviz_chart(dot_str)

        with col2:
            st.markdown("#### Schema Summary")
            st.write("**State Fields:**")
            for sf in spec.state:
                reducer_str = f" `{sf.reducer}`" if sf.reducer != "overwrite" else ""
                st.markdown(f"- `{sf.field}` ({sf.type}){reducer_str}")
            st.write("**Nodes:**")
            for nd in spec.nodes:
                st.markdown(f"- `{nd.id}` — *{nd.type}*")

    with tab_run:
        st.subheader(f"Execute Workflow: `{spec.name}`")
        st.markdown("##### Initial State Inputs")

        # Dynamically build state input form
        initial_input = {}
        with st.form(key=f"run_form_{spec.name}"):
            for field in spec.state:
                default_val = field.default if field.default is not None else ""
                if field.type == "int":
                    initial_input[field.field] = st.number_input(field.field, value=int(default_val) if default_val != "" else 0)
                elif field.type == "float":
                    initial_input[field.field] = st.number_input(field.field, value=float(default_val) if default_val != "" else 0.0)
                elif field.type == "bool":
                    initial_input[field.field] = st.checkbox(field.field, value=bool(default_val))
                elif field.type == "list":
                    initial_input[field.field] = field.default if field.default is not None else []
                elif field.type == "dict":
                    initial_input[field.field] = field.default if field.default is not None else {}
                else:
                    initial_input[field.field] = st.text_input(field.field, value=str(default_val))

            submit_run = st.form_submit_button("▶ Start Workflow Run", type="primary")

        if submit_run:
            thread_id = str(uuid.uuid4())
            st.session_state["active_thread_id"] = thread_id
            st.session_state["active_workflow"] = spec.name
            
            with st.spinner("Executing graph steps..."):
                run_res = engine.run_workflow(spec.name, initial_input, thread_id)
                st.session_state["latest_run"] = run_res

        # Display active run status & state
        if "latest_run" in st.session_state and st.session_state.get("active_workflow") == spec.name:
            run_data = st.session_state["latest_run"]
            t_id = run_data["run_id"]

            st.divider()
            st.markdown(f"### Run Execution: `{t_id[:8]}...`")

            status = run_data["status"]
            if status == "completed":
                st.success("✅ **Status**: Completed")
            elif status == "paused":
                st.warning(f"⏸️ **Status**: Paused at Human Review (`{run_data.get('current_node')}`)")
            elif status == "failed":
                st.error(f"❌ **Status**: Failed — {run_data.get('error')}")

            # State & Audit Logs
            state_val = run_data.get("state", {})
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### Current State")
                st.json(state_val)

            with col_s2:
                st.markdown("#### Audit Logs")
                logs = state_val.get("logs", [])
                if logs:
                    for line in logs:
                        st.markdown(f"- `{line}`")
                else:
                    st.caption("No logs recorded.")

            # Human-in-the-loop Review Form for Paused Run
            if status == "paused":
                st.divider()
                st.markdown("### ✋ Human-in-the-Loop Approval Required")
                st.info("The workflow execution is safely paused in SQLite persistence awaiting your decision.")

                with st.form(key=f"resume_form_{t_id}"):
                    decision_val = st.text_input("Enter decision or approval value:", value="approved")
                    submit_resume = st.form_submit_button("✔ Submit & Resume Graph", type="primary")

                if submit_resume:
                    with st.spinner("Resuming workflow..."):
                        updated_res = engine.resume_workflow(spec.name, t_id, decision_val)
                        st.session_state["latest_run"] = updated_res
                        st.rerun()

    with tab_compile:
        st.subheader("📝 Dynamic YAML Workflow Compiler")
        st.caption("Paste a YAML workflow definition to compile and validate live.")

        default_yaml_snippet = """name: dynamic_custom_workflow
description: Custom user-defined workflow
state:
  - field: message
    type: str
    reducer: overwrite
    default: "Custom Workflow Initialized"
  - field: logs
    type: list
    reducer: append
    default: []
nodes:
  - id: transform
    type: function
    function_name: uppercase_transform
edges:
  - from: START
    to: transform
  - from: transform
    to: END
"""
        yaml_input = st.text_area("YAML Definition", value=default_yaml_snippet, height=300)
        if st.button("Validate & Compile YAML", type="primary"):
            try:
                new_spec = engine.register_yaml(yaml_input)
                st.success(f"Successfully compiled workflow `{new_spec.name}` with {len(new_spec.nodes)} nodes!")
                st.rerun()
            except Exception as e:
                st.error(f"Validation Error: {e}")


if __name__ == "__main__":
    main()
