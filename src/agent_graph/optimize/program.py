"""DSPy module wrapping the dual-audience generation for GEPA.

Two signatures — one per SystemMessage in `dual_audience_node` — so each
GEPA-evolved instruction maps 1:1 onto the live node's clinician / technical
prompt (deployed by editing CLINICIAN_GUIDANCE / TECHNICAL_GUIDANCE in nodes.py).
The rest of the LangGraph pipeline is untouched.
"""
import dspy


class GenerateClinician(dspy.Signature):
    """Given a research question and the retrieved papers, write a clinician
    summary as JSON matching ClinicianSummary. Cite ONLY PMIDs present in the
    retrieved papers; never invent a citation, and never reuse one PMID for two
    different papers."""

    query: str = dspy.InputField(desc="The clinical question")
    papers: str = dspy.InputField(desc="Retrieved papers, each with its PMID")
    clinician_summary: dict = dspy.OutputField(desc="ClinicianSummary as JSON")


class GenerateTechnical(dspy.Signature):
    """Given a research question and the retrieved papers, write a technical
    summary as JSON matching TechnicalSummary. Cite ONLY PMIDs present in the
    retrieved papers; never invent a citation, and never reuse one PMID for two
    different papers."""

    query: str = dspy.InputField(desc="The research question")
    papers: str = dspy.InputField(desc="Retrieved papers, each with its PMID")
    technical_summary: dict = dspy.OutputField(desc="TechnicalSummary as JSON")


class DualAudienceProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.clinician = dspy.Predict(GenerateClinician)
        self.technical = dspy.Predict(GenerateTechnical)

    @staticmethod
    def _format_papers(papers):
        """Render retrieved papers (list[dict]) into prompt text with PMIDs.
        A pre-formatted string passes through unchanged."""
        if isinstance(papers, str):
            return papers
        blocks = []
        for p in papers or []:
            blocks.append(
                f"[PMID {p.get('pmid', '?')}] {p.get('title', '')}\n"
                f"{(p.get('summary') or '')[:400]}\n"
                f"URL: {p.get('url', '')}"
            )
        return "\n\n".join(blocks)

    def forward(self, query, papers):
        text = self._format_papers(papers)
        c = self.clinician(query=query, papers=text)
        t = self.technical(query=query, papers=text)
        return dspy.Prediction(clinician_summary=c.clinician_summary,
                               technical_summary=t.technical_summary)
