"""
Pydantic data models for the Legal Document Reviewer system.
Defines schemas for specialist findings, cross-examination challenges, final findings, and the complete review report.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ClauseFinding(BaseModel):
    """Initial finding raised by a specialist reviewer."""

    clause_reference: str = Field(
        ..., description="Reference or title of the clause (e.g., 'Clause 2: Termination Rights')"
    )
    concern: str = Field(
        ..., description="Detailed explanation of the risk, compliance gap, liability exposure, or obligation"
    )
    initial_severity: int = Field(
        ..., ge=1, le=5, description="Initial severity rating assigned by the specialist (1=low to 5=critical)"
    )
    raised_by: str = Field(
        ..., description="Name of the specialist role that raised this finding (e.g., 'Risk Agent')"
    )


class Challenge(BaseModel):
    """Structured response from a peer specialist during the cross-examination debate round."""

    challenger: str = Field(
        ..., description="Name of the peer specialist reviewing the finding (e.g., 'Compliance Agent')"
    )
    target_clause_reference: str = Field(
        ..., description="The clause reference being evaluated"
    )
    stance: Literal["agree", "dispute"] = Field(
        ..., description="Whether the challenger agrees with the finding/severity or disputes it"
    )
    reasoning: str = Field(
        ..., description="Legal rationale supporting the agreement or dispute"
    )


class FinalFinding(BaseModel):
    """Final finding synthesized by the Judge Agent after considering peer debate."""

    clause_reference: str = Field(
        ..., description="Reference or title of the clause"
    )
    concern: str = Field(
        ..., description="Detailed explanation of the clause issue"
    )
    raised_by: str = Field(
        ..., description="Specialist agent that originally identified the issue"
    )
    final_severity: int = Field(
        ..., ge=1, le=5, description="Final severity score (1-5) assigned by the Judge Agent"
    )
    contested: bool = Field(
        ..., description="True if at least one peer specialist disputed the finding during debate"
    )
    dissent_notes: list[str] = Field(
        default_factory=list, description="List of reasoning notes from disputing peer specialists"
    )


class ReviewReport(BaseModel):
    """Complete summary report produced by the Judge Agent."""

    document_name: str = Field(
        ..., description="Name or identifier of the analyzed contract document"
    )
    findings: list[FinalFinding] = Field(
        default_factory=list, description="Synthesized list of final clause findings"
    )
    overall_risk_summary: str = Field(
        ..., description="Executive summary of overall document risk profile and recommendations"
    )
    debate_transcript: list[Challenge] = Field(
        default_factory=list, description="Full log of cross-examination challenges from the debate round"
    )
