import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import streamlit as st

load_dotenv()

# ── Pydantic models — structured LLM output ──────────────────────
class ResearchPlan(BaseModel):
    title: str = Field(description="A clear title for the research topic")
    subtopics: list[str] = Field(
        description="3 to 5 focused subtopics that together answer the question"
    )
    rationale: str = Field(
        description="One sentence explaining why these subtopics cover the question"
    )

class SubtopicFindings(BaseModel):
    subtopic: str = Field(description="The subtopic that was researched")
    summary: str = Field(description="Detailed findings, 150-200 words")
    key_points: list[str] = Field(description="3 to 5 key facts as bullet points")
    confidence: float = Field(
        description="Confidence score 0.0 to 1.0 based on certainty of information",
        ge=0.0, le=1.0
    )

class FinalReport(BaseModel):
    title: str = Field(description="Title of the final research report")
    executive_summary: str = Field(description="2-3 sentence overview of all findings")
    key_findings: list[str] = Field(description="5 most important findings across all subtopics")
    limitations: str = Field(description="What this research doesn't cover or where uncertainty exists")
    overall_confidence: float = Field(
        description="Overall confidence 0.0 to 1.0 across all findings",
        ge=0.0, le=1.0
    )


# ── Agent functions ───────────────────────────────────────────────
def get_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def plan_research(question: str) -> ResearchPlan:
    llm = get_llm().with_structured_output(ResearchPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research planner. Break the question into 3-5 focused subtopics "
         "that together comprehensively answer it. Be specific and non-overlapping."),
        ("user", "Research question: {question}")
    ])
    return (prompt | llm).invoke({"question": question})


def research_subtopic(question: str, subtopic: str) -> SubtopicFindings:
    llm = get_llm(temperature=0.2).with_structured_output(SubtopicFindings)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert researcher. Research the subtopic in the context of "
         "the main question. Provide detailed findings, 3-5 key points, and a "
         "confidence score (0.0-1.0) reflecting how well-established the information is. "
         "Score 0.9+ for well-known facts, 0.6-0.8 for generally accepted knowledge, "
         "below 0.6 for emerging or debated topics."),
        ("user",
         "Main question: {question}\n"
         "Subtopic to research: {subtopic}")
    ])
    return (prompt | llm).invoke({"question": question, "subtopic": subtopic})


def synthesize_report(
    question: str,
    plan: ResearchPlan,
    findings: list[SubtopicFindings]
) -> FinalReport:
    llm = get_llm(temperature=0.2).with_structured_output(FinalReport)
    findings_text = "\n\n".join([
        f"Subtopic: {f.subtopic}\n"
        f"Summary: {f.summary}\n"
        f"Key Points: {chr(10).join('- ' + p for p in f.key_points)}\n"
        f"Confidence: {f.confidence:.0%}"
        for f in findings
    ])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research synthesizer. Create a comprehensive final report "
         "from the research findings. Extract the 5 most important insights, "
         "write a clear executive summary, note limitations, and give an "
         "overall confidence score."),
        ("user",
         "Original question: {question}\n\n"
         "Research findings:\n{findings}")
    ])
    return (prompt | llm).invoke({"question": question, "findings": findings_text})


def build_markdown_report(
    question: str,
    plan: ResearchPlan,
    findings: list[SubtopicFindings],
    report: FinalReport
) -> str:
    sections = "\n\n".join([
        f"### {f.subtopic}\n\n"
        f"{f.summary}\n\n"
        f"**Key Points:**\n"
        + "\n".join(f"- {p}" for p in f.key_points)
        + f"\n\n**Confidence:** {f.confidence:.0%}"
        for f in findings
    ])
    return f"""# {report.title}

**Research Question:** {question}

---

## Executive Summary

{report.executive_summary}

---

## Research Plan

{plan.rationale}

**Subtopics covered:**
{chr(10).join(f'- {s}' for s in plan.subtopics)}

---

## Detailed Findings

{sections}

---

## Key Findings

{chr(10).join(f'- {k}' for k in report.key_findings)}

---

## Limitations

{report.limitations}

---

**Overall Confidence:** {report.overall_confidence:.0%}
"""


# ── Streamlit UI ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Agentic Research Assistant")
st.caption("CalderR Internship 2026 · Project 1-P-C")
st.markdown(
    "Enter a research question. The agent will plan a strategy, "
    "research each subtopic sequentially, and synthesize a structured report."
)

question = st.text_input(
    "Research Question",
    placeholder="e.g. What are the key challenges in deploying LLMs in production?"
)

if st.button("Start Research", type="primary", disabled=not question):

    # ── Step 1: Plan ──────────────────────────────────────────────
    with st.status("📋 Planning research strategy...", expanded=True) as status:
        st.write("Breaking your question into focused subtopics...")
        try:
            plan = plan_research(question)
            st.write(f"✓ Identified **{len(plan.subtopics)}** subtopics to research")
            status.update(label="📋 Research plan ready", state="complete")
        except Exception as e:
            status.update(label=f"Planning failed: {e}", state="error")
            st.stop()

    with st.expander("📋 Research Plan", expanded=True):
        st.subheader(plan.title)
        st.write(plan.rationale)
        for i, subtopic in enumerate(plan.subtopics, 1):
            st.write(f"**{i}.** {subtopic}")

    # ── Step 2: Research loop ─────────────────────────────────────
    all_findings: list[SubtopicFindings] = []
    progress_bar = st.progress(0, text="Starting research loop...")

    for i, subtopic in enumerate(plan.subtopics):
        with st.status(f"🔍 Researching: {subtopic}...", expanded=False) as status:
            try:
                findings = research_subtopic(question, subtopic)
                all_findings.append(findings)

                st.write(findings.summary)
                st.write("**Key Points:**")
                for point in findings.key_points:
                    st.write(f"• {point}")
                st.metric("Confidence", f"{findings.confidence:.0%}")

                status.update(
                    label=f"✓ {subtopic[:50]} — {findings.confidence:.0%} confidence",
                    state="complete"
                )
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")

        progress_bar.progress(
            (i + 1) / len(plan.subtopics),
            text=f"Researched {i + 1} of {len(plan.subtopics)} subtopics"
        )

    # ── Step 3: Synthesize ────────────────────────────────────────
    with st.status("⚗️ Synthesizing final report...", expanded=True) as status:
        st.write("Combining all findings into a structured report...")
        try:
            report = synthesize_report(question, plan, all_findings)
            status.update(label="✓ Report complete", state="complete")
        except Exception as e:
            status.update(label=f"Synthesis failed: {e}", state="error")
            st.stop()

    # ── Final report ──────────────────────────────────────────────
    st.divider()
    st.header(f"📄 {report.title}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Executive Summary")
        st.write(report.executive_summary)
    with col2:
        confidence_pct = report.overall_confidence
        st.metric(
            "Overall Confidence",
            f"{confidence_pct:.0%}",
            delta="high" if confidence_pct >= 0.7 else "medium" if confidence_pct >= 0.4 else "low"
        )

    st.subheader("Key Findings")
    for finding in report.key_findings:
        st.write(f"✦ {finding}")

    st.subheader("Confidence by Subtopic")
    chart_data = {
        f.subtopic[:35] + ("..." if len(f.subtopic) > 35 else ""): f.confidence
        for f in all_findings
    }
    st.bar_chart(chart_data, y_label="Confidence", use_container_width=True)

    st.subheader("Limitations")
    st.info(report.limitations)

    # ── Download ──────────────────────────────────────────────────
    markdown_report = build_markdown_report(question, plan, all_findings, report)
    st.download_button(
        label="⬇ Download Report (Markdown)",
        data=markdown_report,
        file_name=f"research_{'_'.join(question.split()[:5])}.md",
        mime="text/markdown",
    )
