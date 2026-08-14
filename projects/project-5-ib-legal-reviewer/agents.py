"""
Multi-agent legal document review agents and workflow facilitator.
Includes specialist review agents (Risk, Compliance, Liability, Obligations),
a debate facilitator dispatcher, and a Judge agent.
Supports deterministic offline Stub mode and online LLM mode (--real) via ChatGroq.
"""

import os
from typing import Literal
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

try:
    # pyrefly: ignore [missing-import]
    from models import (
        ClauseFinding,
        Challenge,
        FinalFinding,
        ReviewReport,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from project_5_ib_legal_reviewer.models import (
        ClauseFinding,
        Challenge,
        FinalFinding,
        ReviewReport,
    )

SPECIALIST_ROLES = [
    "Risk Agent",
    "Compliance Agent",
    "Liability Agent",
    "Obligations Agent",
]


class SpecialistFindingsOutput(BaseModel):
    """Wrapper for structured output from specialist review agents in real mode."""
    findings: list[ClauseFinding] = Field(
        default_factory=list, description="List of clause findings identified by the specialist"
    )


class JudgeOutput(BaseModel):
    """Wrapper for structured output from the Judge agent in real mode."""
    findings: list[FinalFinding] = Field(
        default_factory=list, description="Final clause findings synthesized after debate"
    )
    overall_risk_summary: str = Field(
        ..., description="Executive risk summary of the contract document"
    )


# ---------------------------------------------------------------------------
# STUB MODE IMPLEMENTATION
# ---------------------------------------------------------------------------

def _stub_specialist_review(document_text: str, role: str) -> list[ClauseFinding]:
    """
    Deterministic specialist review based on contract content and keywords.
    Each specialist looks for specific flaggable patterns.
    """
    findings: list[ClauseFinding] = []
    text_lower = document_text.lower()

    if role == "Risk Agent":
        if "sole discretion" in text_lower or "unilaterally" in text_lower:
            ref = "Clause 4: Injunctive Relief & Discretion" if "injunctive" in text_lower else "Clause 1: Unilateral Modifications"
            if "injunctive" in text_lower:
                concern = "Disclosing party retains sole discretion to seek immediate injunctive relief without bond, creating severe procedural imbalance."
                severity = 5
            elif "services agreement" in text_lower or "independent consulting" in text_lower:
                ref = "Clause 1: Scope of Work"
                concern = "Client reserves right to unilaterally modify deliverables and timelines without Consultant consent or rate adjustments."
                severity = 4
            else:
                concern = "Provider reserves right to modify terms, features, and pricing in sole discretion without notice."
                severity = 4
            findings.append(ClauseFinding(clause_reference=ref, concern=concern, initial_severity=severity, raised_by=role))

        if "without cause" in text_lower or "immediate written notice" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 2: Termination Rights",
                concern="Asymmetric termination provision allowing Client immediate termination without cause while binding contractor to 90 days notice.",
                initial_severity=5,
                raised_by=role,
            ))

    elif role == "Compliance Agent":
        if "under any circumstances" in text_lower and "subpoena" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 2: Obligations of Non-Disclosure",
                concern="Confidentiality restriction lacks statutory whistleblower and legal subpoena carve-outs, violating regulatory reporting protections.",
                initial_severity=4,
                raised_by=role,
            ))

        if "personal computers" in text_lower or "physical workspaces" in text_lower or "unannounced audits" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 5: System Audit and Access",
                concern="Unannounced physical audit rights over personal computers and workspaces without cause violate privacy standards and statutory employee rights.",
                initial_severity=4,
                raised_by=role,
            ))

        if "cayman islands" in text_lower or "class action" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 4: Governing Law and Exclusive Forum",
                concern="Offshore jurisdiction (Cayman Islands) and mandatory class action waiver present regulatory enforcement obstacles.",
                initial_severity=3,
                raised_by=role,
            ))

    elif role == "Liability Agent":
        if "perpetually" in text_lower or "non-solicitation" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 3: Perpetual Non-Solicitation",
                concern="Worldwide perpetual non-solicit clause is an unreasonable restraint of trade with extreme lingering liability exposure.",
                initial_severity=4,
                raised_by=role,
            ))

        if "unlimited" in text_lower or "indemnify" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 4: Indemnification and Unlimited Liability",
                concern="Consultant assumes unlimited liability and unilateral indemnification burden while Client completely disclaims all liability.",
                initial_severity=5,
                raised_by=role,
            ))

        if "as is" in text_lower or "zero liability" in text_lower or "data corruption" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 2: Disclaimer of Warranties and Zero Liability",
                concern="Complete disclaimer of warranties and zero liability for data corruption or security breaches leaves customer completely unprotected.",
                initial_severity=5,
                raised_by=role,
            ))

    elif role == "Obligations Agent":
        if "ownership transfers immediately regardless" in text_lower or "upon creation" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 3: Intellectual Property Assignment",
                concern="IP ownership transfers to Client immediately upon creation prior to receipt of payment, risking unpaid work seizure.",
                initial_severity=4,
                raised_by=role,
            ))

        if "within 24 hours" in text_lower:
            findings.append(ClauseFinding(
                clause_reference="Clause 3: Mandatory 24-Hour Notice Period",
                concern="24-hour mandatory notice window for billing and defect claims is an oppressive operational burden causing automatic waiver of rights.",
                initial_severity=5,
                raised_by=role,
            ))

    return findings


def _stub_peer_challenge(finding: ClauseFinding, challenger_role: str) -> Challenge:
    """
    Deterministic cross-examination challenge from a peer specialist.
    Produces realistic agreement or genuine disputes with reasoning.
    """
    ref = finding.clause_reference.lower()

    # Rule 1: Compliance Agent disputes perpetual non-solicit severity raised by Liability Agent
    if challenger_role == "Compliance Agent" and "non-solicitation" in ref:
        return Challenge(
            challenger=challenger_role,
            target_clause_reference=finding.clause_reference,
            stance="dispute",
            reasoning="Perpetual non-solicit clauses are routinely struck down as void and unenforceable restraints of trade in most jurisdictions. The actual legal risk is lower than initial severity 4; recommending severity 2.",
        )

    # Rule 2: Risk Agent disputes Compliance Agent's system audit severity
    if challenger_role == "Risk Agent" and "system audit" in ref:
        return Challenge(
            challenger=challenger_role,
            target_clause_reference=finding.clause_reference,
            stance="dispute",
            reasoning="While unannounced audits pose privacy concerns, in commercial software contracting audit rights are standard operational controls. Initial severity 4 is overstated; recommending severity 3.",
        )

    # Rule 3: Compliance Agent disputes Obligations Agent's 24-hour notice period severity
    if challenger_role == "Compliance Agent" and "24-hour notice" in ref:
        return Challenge(
            challenger=challenger_role,
            target_clause_reference=finding.clause_reference,
            stance="dispute",
            reasoning="While 24 hours is a tight operational deadline, notice windows in SaaS agreements do not violate statutory regulations. Initial severity 5 is overstated for an administrative clause; recommending severity 3.",
        )

    # Rule 4: Liability Agent disputes Risk Agent's unilateral modification severity in SaaS
    if challenger_role == "Liability Agent" and "unilateral modification" in ref:
        return Challenge(
            challenger=challenger_role,
            target_clause_reference=finding.clause_reference,
            stance="dispute",
            reasoning="Unilateral modification clauses are standard in SaaS terms of service provided customers can terminate. Severity 4 is high; recommending severity 3.",
        )

    # Default stance: Peer agrees with the finding and initial severity
    return Challenge(
        challenger=challenger_role,
        target_clause_reference=finding.clause_reference,
        stance="agree",
        reasoning=f"{challenger_role} concurs that {finding.clause_reference} presents genuine operational or legal risk as identified.",
    )


def _stub_judge_synthesis(
    document_name: str,
    findings: list[ClauseFinding],
    challenges: list[Challenge],
) -> ReviewReport:
    """
    Synthesizes final findings and adjusts severity based on peer challenges.
    Findings with any dispute stance are marked contested=True with non-empty dissent_notes.
    Final severity is modified if disputed.
    """
    final_findings: list[FinalFinding] = []

    for f in findings:
        # Get all challenges targeted at this clause
        clause_challenges = [
            c for c in challenges if c.target_clause_reference == f.clause_reference
        ]
        disputes = [c for c in clause_challenges if c.stance == "dispute"]
        is_contested = len(disputes) > 0
        dissent_notes = [f"[{c.challenger}] {c.reasoning}" for c in disputes]

        # Calculate final severity responsive to debate
        final_sev = f.initial_severity
        if is_contested:
            # Check if dissenters argued for lower severity
            for c in disputes:
                if "recommending severity 2" in c.reasoning:
                    final_sev = 2
                elif "recommending severity 3" in c.reasoning:
                    final_sev = 3
                elif final_sev > 1:
                    final_sev -= 1

        final_findings.append(
            FinalFinding(
                clause_reference=f.clause_reference,
                concern=f.concern,
                raised_by=f.raised_by,
                final_severity=final_sev,
                contested=is_contested,
                dissent_notes=dissent_notes,
            )
        )

    summary = (
        f"Legal Review Summary for '{document_name}': Identified {len(final_findings)} total findings "
        f"({sum(1 for f in final_findings if f.contested)} contested during cross-examination debate). "
        f"Critical issues center on asymmetric liability, termination terms, and non-solicitation scope."
    )

    return ReviewReport(
        document_name=document_name,
        findings=final_findings,
        overall_risk_summary=summary,
        debate_transcript=challenges,
    )


# ---------------------------------------------------------------------------
# REAL MODE IMPLEMENTATION (ChatGroq)
# ---------------------------------------------------------------------------

def _get_groq_llm():
    """Returns ChatGroq instance or raises clean RuntimeError if GROQ_API_KEY is missing."""
    try:
        # pyrefly: ignore [missing-import]
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required for --real mode.")
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def _real_specialist_review(document_text: str, role: str) -> list[ClauseFinding]:
    """Invokes LLM specialist to independently extract clause findings."""
    llm = _get_groq_llm()

    role_prompts = {
        "Risk Agent": "You are a Senior Contract Risk Specialist. Identify unfavorable business terms, one-sided rights, and missing protections in the contract.",
        "Compliance Agent": "You are a Regulatory Compliance Attorney. Check for statutory red flags, missing legal carve-outs, and privacy/labor compliance violations.",
        "Liability Agent": "You are a Legal Liability Expert. Map liability exposures, indemnity imbalances, and un-capped financial risk.",
        "Obligations Agent": "You are a Contractual Obligations Specialist. Extract strict performance deadlines, unfair operational burdens, and audit requirements.",
    }

    prompt = f"""{role_prompts.get(role, "")}

Analyze the following contract document and extract 1-3 critical findings.
For each finding, provide:
- clause_reference: exact header or clause title
- concern: detailed explanation of the issue
- initial_severity: integer rating from 1 (low) to 5 (critical)
- raised_by: set to "{role}"

Contract Text:
{document_text}
"""
    structured_llm = llm.with_structured_output(SpecialistFindingsOutput)
    res = structured_llm.invoke(prompt)
    findings = res.findings if res and res.findings else []
    for f in findings:
        f.raised_by = role
    return findings


def _real_peer_challenge(finding: ClauseFinding, challenger_role: str, document_text: str) -> Challenge:
    """Invokes LLM to cross-examine a peer's finding."""
    llm = _get_groq_llm()

    prompt = f"""You are the {challenger_role}. A peer specialist ({finding.raised_by}) raised the following finding on a contract:

Target Clause: {finding.clause_reference}
Concern: {finding.concern}
Initial Severity Rating: {finding.initial_severity} / 5

Contract Context:
{document_text}

Evaluate this finding critically from your role perspective ({challenger_role}):
- Decide whether you 'agree' or 'dispute' the finding or its severity rating. Be rigorous — dispute if the severity is overstated, if the risk is standard industry practice, or if your legal discipline interprets it differently.
- Provide clear, professional legal reasoning.

Set challenger to "{challenger_role}", target_clause_reference to "{finding.clause_reference}", stance to "agree" or "dispute", and reasoning.
"""
    structured_llm = llm.with_structured_output(Challenge)
    res = structured_llm.invoke(prompt)
    res.challenger = challenger_role
    res.target_clause_reference = finding.clause_reference
    return res


def _real_judge_synthesis(
    document_name: str,
    findings: list[ClauseFinding],
    challenges: list[Challenge],
    document_text: str,
) -> ReviewReport:
    """Invokes Judge Agent LLM to synthesize final report and set final severities."""
    llm = _get_groq_llm()

    findings_text = "\n".join(
        [
            f"- [{f.raised_by}] {f.clause_reference} (Initial Sev: {f.initial_severity}): {f.concern}"
            for f in findings
        ]
    )

    challenges_text = "\n".join(
        [
            f"- [{c.challenger} -> {c.target_clause_reference}] Stance: {c.stance.upper()} | Reasoning: {c.reasoning}"
            for c in challenges
        ]
    )

    prompt = f"""You are the Senior Chief Legal Judge Agent. You must review initial specialist findings and peer debate challenges for the contract document '{document_name}'.

Initial Specialist Findings:
{findings_text}

Peer Cross-Examination Debate Transcript:
{challenges_text}

Contract Document Text:
{document_text}

Your Task:
1. Synthesize a final list of findings. For each finding:
   - Assign final_severity (1-5). If peer specialists raised valid disputes or argued severity was overstated/understated, you MUST adjust final_severity accordingly.
   - Set contested to true if ANY peer challenged the finding with 'dispute', otherwise false.
   - Include all disputing reasoning in dissent_notes.
2. Produce an overall executive risk summary for the document.
"""
    structured_llm = llm.with_structured_output(JudgeOutput)
    res = structured_llm.invoke(prompt)

    return ReviewReport(
        document_name=document_name,
        findings=res.findings if res else [],
        overall_risk_summary=res.overall_risk_summary if res else "Review complete.",
        debate_transcript=challenges,
    )


# ---------------------------------------------------------------------------
# MAIN WORKFLOW CONTROLLER
# ---------------------------------------------------------------------------

def run_legal_review(
    document_name: str, document_text: str, real_mode: bool = False
) -> ReviewReport:
    """
    Main orchestration flow for the Legal Document Reviewer system:
    1. Ingestion: contract text received.
    2. Parallel Independent Review: all 4 specialists review document independently.
    3. Debate Round (Facilitator): routes findings to 3 non-raising peer specialists.
    4. Judge Agent: synthesizes final report and severity scores.
    """
    # 2. Parallel Independent Review
    initial_findings: list[ClauseFinding] = []
    for role in SPECIALIST_ROLES:
        if real_mode:
            role_findings = _real_specialist_review(document_text, role)
        else:
            role_findings = _stub_specialist_review(document_text, role)
        initial_findings.extend(role_findings)

    # 3. Debate Round (Debate Facilitator Dispatcher)
    debate_transcript: list[Challenge] = []
    for finding in initial_findings:
        # Route to the 3 specialists who did NOT raise it
        peers = [r for r in SPECIALIST_ROLES if r != finding.raised_by]
        for peer_role in peers:
            if real_mode:
                challenge = _real_peer_challenge(finding, peer_role, document_text)
            else:
                challenge = _stub_peer_challenge(finding, peer_role)
            debate_transcript.append(challenge)

    # 4. Judge Agent Synthesis
    if real_mode:
        report = _real_judge_synthesis(
            document_name, initial_findings, debate_transcript, document_text
        )
    else:
        report = _stub_judge_synthesis(
            document_name, initial_findings, debate_transcript
        )

    return report
