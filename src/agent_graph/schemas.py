"""Pydantic output schemas for the two-audience structured-output node."""
from pydantic import BaseModel, Field
from typing import Literal


class Evidence(BaseModel):
    claim: str
    pmid: str
    source_url: str


class ClinicianSummary(BaseModel):
    audience: Literal["clinician"] = "clinician"
    bottom_line: str = Field(..., description="One-sentence actionable takeaway")
    key_findings: list[str]
    evidence: list[Evidence]
    confidence_note: str = Field(..., description="What is and isn't well-supported")


class TechnicalSummary(BaseModel):
    audience: Literal["technical"] = "technical"
    detailed_findings: str
    methodology_notes: str
    evidence: list[Evidence]
    caveats: list[str]
