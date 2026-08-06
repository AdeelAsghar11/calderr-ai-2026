"""
Streamlit interactive dashboard for Project 5-I-B: Multi-Agent Legal Document Reviewer.
Provides contract selection/upload, inline clause visualization, contested findings callouts,
and peer debate cross-examination inspection.
"""

import sys
import os
from pathlib import Path
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to sys.path
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from agents import run_legal_review, SPECIALIST_ROLES
    from models import ReviewReport
except ImportError:
    from project_5_ib_legal_reviewer.agents import run_legal_review, SPECIALIST_ROLES
    from project_5_ib_legal_reviewer.models import ReviewReport

st.set_page_config(
    page_title="Multi-Agent Legal Document Reviewer",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Multi-Agent Legal Document Reviewer")
st.markdown(
    "A portfolio-grade legal risk assessment platform utilizing independent specialist agents "
    "(Risk, Compliance, Liability, Obligations), a cross-examination debate facilitator, and a Judge Agent."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio(
    "Execution Mode",
    ["Stub Mode (Offline Deterministic)", "Real Mode (Groq Llama-3.3-70B)"],
    index=0,
)
real_mode = mode.startswith("Real")

if real_mode and not os.getenv("GROQ_API_KEY"):
    st.sidebar.error("⚠️ GROQ_API_KEY environment variable is missing!")

st.sidebar.subheader("📄 Document Source")
input_source = st.sidebar.radio("Select Contract Input", ["Sample Contracts", "Upload Contract File"])

contract_text = ""
document_name = ""

sample_dir = PROJECT_DIR / "sample_contracts"
if input_source == "Sample Contracts":
    sample_files = sorted(list(sample_dir.glob("*.txt")))
    file_map = {f.name: f for f in sample_files}
    selected_name = st.sidebar.selectbox("Choose a Sample Contract", list(file_map.keys()))
    if selected_name:
        document_name = selected_name
        with open(file_map[selected_name], "r", encoding="utf-8") as f:
            contract_text = f.read()
else:
    uploaded_file = st.sidebar.file_uploader("Upload a Contract (.txt)", type=["txt"])
    if uploaded_file:
        document_name = uploaded_file.name
        contract_text = uploaded_file.read().decode("utf-8")

if not contract_text:
    st.info("👈 Select a sample contract or upload a text file in the sidebar to begin analysis.")
    st.stop()

# Layout: Split Contract Document & Review Report
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📜 Document Text: {document_name}")
    st.text_area("Contract Content", contract_text, height=520, disabled=True)

with col2:
    st.subheader("🔍 Legal Review & Cross-Examination")
    if st.button("🚀 Run Multi-Agent Legal Review", type="primary"):
        with st.spinner("Executing specialist reviews, debate round, and Judge synthesis..."):
            try:
                report = run_legal_review(document_name, contract_text, real_mode=real_mode)
                st.session_state["current_report"] = report
            except Exception as e:
                st.error(f"Execution Error: {e}")

if "current_report" in st.session_state:
    report: ReviewReport = st.session_state["current_report"]

    st.markdown("---")
    st.header("📊 Executive Risk Summary")
    st.warning(report.overall_risk_summary)

    # Metrics Summary
    total_findings = len(report.findings)
    contested_count = sum(1 for f in report.findings if f.contested)
    avg_severity = (
        sum(f.final_severity for f in report.findings) / total_findings if total_findings else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Findings", total_findings)
    m2.metric("Contested Findings", contested_count, delta=f"{contested_count} challenged", delta_color="inverse")
    m3.metric("Average Severity", f"{avg_severity:.1f} / 5")
    m4.metric("Debate Transcript Size", f"{len(report.debate_transcript)} challenges")

    # Tabs for Detailed Findings & Debate Transcript
    tab1, tab2 = st.tabs(["⚖️ Synthesized Findings (Judge Agent)", "💬 Debate Cross-Examination Transcript"])

    with tab1:
        st.subheader("Clause Findings & Severity Ratings")
        for idx, finding in enumerate(report.findings, start=1):
            sev = finding.final_severity
            sev_badge = "🔴 Critical" if sev >= 5 else ("🟠 High" if sev == 4 else ("🟡 Moderate" if sev == 3 else "🟢 Low"))

            if finding.contested:
                st.error(
                    f"**#{idx}. {finding.clause_reference}** | Severity: **{sev}/5** ({sev_badge}) | "
                    f"Raised By: `{finding.raised_by}` | ⚠️ **CONTESTED IN DEBATE**"
                )
            else:
                st.success(
                    f"**#{idx}. {finding.clause_reference}** | Severity: **{sev}/5** ({sev_badge}) | "
                    f"Raised By: `{finding.raised_by}` | ✅ Uncontested"
                )

            st.write(f"**Concern:** {finding.concern}")

            if finding.contested and finding.dissent_notes:
                st.markdown("**Dissent & Challenge Notes from Debate Round:**")
                for note in finding.dissent_notes:
                    st.info(f"💬 {note}")
            st.markdown("---")

    with tab2:
        st.subheader("Peer Specialist Cross-Examination Log")
        for challenge in report.debate_transcript:
            stance_icon = "🔴 DISPUTE" if challenge.stance == "dispute" else "🟢 AGREE"
            with st.expander(
                f"{challenge.challenger} -> {challenge.target_clause_reference} [{stance_icon}]"
            ):
                st.write(f"**Target Clause:** {challenge.target_clause_reference}")
                st.write(f"**Peer Challenger:** {challenge.challenger}")
                st.write(f"**Stance:** {challenge.stance.upper()}")
                st.write(f"**Legal Rationale:** {challenge.reasoning}")
