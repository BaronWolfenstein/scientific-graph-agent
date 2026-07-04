"""Faithfulness evaluation: checks that summary claims are grounded in retrieved papers."""
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from agent_graph.llm import get_llm


class ClaimList(BaseModel):
    claims: list[str] = Field(..., description="Atomic factual claims extracted from the summary")


class ClaimVerification(BaseModel):
    supported: bool = Field(..., description="True if the claim is supported by the provided context")


class FaithfulnessScore(BaseModel):
    supported_fraction: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of the summary's factual claims directly supported by the papers")
    reasoning: str = Field(..., description="Brief justification for the score")


def compute_faithfulness_single(summary: str, papers: list, llm=None):
    """Single-LLM-call faithfulness estimate — the cheaper GEPA-metric variant of
    `compute_faithfulness` (which makes 1 + N-claim calls). Returns
    (score, 0, 0) to match the tuple shape. `llm` is injectable for testing.
    """
    if not summary or not papers:
        return 1.0, 0, 0

    llm = llm or get_llm(temperature=0, max_tokens=1024)
    context = "\n\n".join(
        f"[Paper {i}] {p.get('title', '')}\n{(p.get('summary') or '')[:600]}"
        for i, p in enumerate(papers, 1)
    )
    result = llm.with_structured_output(FaithfulnessScore).invoke([
        SystemMessage(content=(
            "Estimate what fraction of the summary's factual claims are DIRECTLY "
            "supported by the provided papers. Judge only on what the papers state; "
            "unsupported or extrapolated claims lower the fraction. "
            "Return supported_fraction in [0,1]."
        )),
        HumanMessage(content=f"Summary:\n{summary}\n\nPapers:\n{context}"),
    ])
    return float(result.supported_fraction), 0, 0


def _extract_claims(summary: str, llm) -> list[str]:
    structured = llm.with_structured_output(ClaimList)
    result = structured.invoke([
        SystemMessage(content=(
            "Extract every atomic factual claim from this scientific summary. "
            "Each claim must be a single verifiable statement. "
            "Ignore formatting, headers, and citation markers like [Paper 1]."
        )),
        HumanMessage(content=summary),
    ])
    return result.claims


def _verify_claim(claim: str, context: str, llm) -> bool:
    structured = llm.with_structured_output(ClaimVerification)
    result = structured.invoke([
        SystemMessage(content=(
            "Decide whether the claim is supported by the context. "
            "Base your judgment only on what is explicitly stated or directly implied in the context. "
            "If the context does not address the claim, answer false."
        )),
        HumanMessage(content=f"Claim: {claim}\n\nContext:\n{context}"),
    ])
    return result.supported


def compute_faithfulness(
    summary: str,
    papers: list[dict],
) -> tuple[float, int, int]:
    """
    Compute faithfulness: fraction of summary claims supported by the retrieved papers.

    Uses two LLM calls: one to extract claims, one per claim to verify.

    Args:
        summary: The generated summary text.
        papers: List of paper dicts with 'title' and 'summary' (abstract) fields.

    Returns:
        (score, n_supported, n_total)
        score is n_supported / n_total, or 1.0 if no claims were extracted.
    """
    if not summary or not papers:
        return 1.0, 0, 0

    llm = get_llm(temperature=0, max_tokens=4096)

    context = "\n\n".join(
        f"[Paper {i}] {p.get('title', '')}\n{p.get('summary', '')[:600]}"
        for i, p in enumerate(papers, 1)
    )

    claims = _extract_claims(summary, llm)
    if not claims:
        return 1.0, 0, 0

    n_supported = sum(_verify_claim(c, context, llm) for c in claims)
    score = n_supported / len(claims)
    return score, n_supported, len(claims)
