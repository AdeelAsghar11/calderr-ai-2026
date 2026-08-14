"""
Streamlit Dashboard for AI Software Engineering Platform.

Provides UI for feature spec input, stage progress tracking, typed contract inspection,
QA and Security audit summaries, and honest Docker validation status reporting.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import streamlit as st
from typing import Optional

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
from src.pipeline import run_full_pipeline
# pyrefly: ignore [missing-import]
from src.schema import PipelineRun


def render_dashboard_view(pipeline_run: PipelineRun):
    """
    Renders the detailed results view for a PipelineRun artifact.

    Why Docker absence is rendered explicitly as 'not run (Docker unavailable)':
    Honesty in portfolio artifacts requires accurately representing what was verified vs skipped.
    """
    st.subheader(f"🚀 Execution Run: `{pipeline_run.run_id}`")
    st.caption(f"Feature Spec: {pipeline_run.feature_spec}")

    # Stage Completion Badges
    st.markdown("**Completed Pipeline Stages:**")
    st.info(" -> ".join([s.upper() for s in pipeline_run.stages_completed]))

    # Contract View
    st.markdown("### 📜 System Contract")
    st.write(f"**Feature Identifier:** `{pipeline_run.system_contract.feature_name}`")

    endpoint_rows = []
    for ep in pipeline_run.system_contract.endpoints:
        endpoint_rows.append(
            {
                "Method": ep.method,
                "Path": ep.path,
                "Description": ep.description,
                "Request Fields": str(ep.request_fields),
                "Response Fields": str(ep.response_fields),
            }
        )
    st.table(endpoint_rows)

    # QA & Security Audit Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 QA Audit Report")
        qa = pipeline_run.qa_report
        st.metric(
            label="Tests Passed / Written",
            value=f"{qa.tests_passed} / {qa.tests_written}",
            delta=f"{qa.tests_failed} Failed",
            delta_color="inverse" if qa.tests_failed > 0 else "normal",
        )
        for r in qa.results:
            status_icon = "✅" if r.passed else "❌"
            st.text(f"{status_icon} {r.test_name}")

    with col2:
        st.markdown("### 🛡️ Security Scan Report")
        sec = pipeline_run.security_report
        st.metric(
            label="Critical Findings",
            value=sec.critical_count,
            delta=f"{len(sec.findings)} Total Findings",
            delta_color="inverse" if sec.critical_count > 0 else "normal",
        )
        if sec.findings:
            for f in sec.findings:
                st.warning(f"**[{f.severity.upper()}]** {f.category} - {f.location}\n{f.description}")
        else:
            st.success("No security vulnerabilities detected.")

    # Docker DevOps & Validation Status
    st.markdown("### 🐳 Docker Container & Runtime Validation")
    if pipeline_run.docker_build_result is None or pipeline_run.validation_report is None:
        st.warning("⚠️ **Status:** Not run (Docker unavailable on host system)")
    else:
        db = pipeline_run.docker_build_result
        val = pipeline_run.validation_report

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.write(f"**Image Tag:** `{db.image_tag}`")
            st.write(f"**Build Status:** {'✅ Succeeded' if db.build_succeeded else '❌ Failed'}")
            st.write(f"**Build Time:** {db.build_duration_seconds}s")
        with d_col2:
            st.write(f"**Container Started:** {'✅ Yes' if val.container_started else '❌ No'}")
            st.write(f"**Validation Overall:** {'✅ Passed' if val.all_passed else '❌ Failed'}")
            st.write(f"**Teardown Clean:** {'✅ Yes' if val.teardown_succeeded else '❌ No'}")


def main():
    st.set_page_config(
        page_title="AI Software Engineering Platform",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Production AI Software Engineering Platform")
    st.caption("End-to-End Autonomous Pipeline: PM → Architect → Codegen → Integration → QA → Security → DevOps → Validation")

    spec_input = st.text_area(
        "Enter Feature Specification:",
        value="A simple todo list API: users can add a todo, list all todos, mark a todo complete, and delete a todo.",
        height=100,
    )

    if st.button("🚀 Run Pipeline", type="primary"):
        with st.spinner("Executing autonomous multi-agent pipeline..."):
            output_dir = os.path.abspath("generated_runs/dashboard_run")
            pipeline_run = run_full_pipeline(
                feature_spec=spec_input,
                output_dir=output_dir,
                use_real=False,
            )
            st.success("Pipeline Execution Completed!")
            render_dashboard_view(pipeline_run)


if __name__ == "__main__":
    main()
