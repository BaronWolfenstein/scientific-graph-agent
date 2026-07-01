"""DSPy module wrapping ONLY the dual-audience generation step for GEPA.

The rest of the LangGraph pipeline is untouched. The instruction GEPA evolves for
`GenerateDualAudience` is exported by `run_gepa.py` and pasted back into
`dual_audience_node` (or loaded as a prompt asset).
"""
import dspy


class GenerateDualAudience(dspy.Signature):
    """Given a research question and the retrieved papers, write a clinician
    summary and a technical summary as JSON matching ClinicianSummary /
    TechnicalSummary. Cite ONLY PMIDs present in the retrieved papers; never
    invent a citation."""

    query: str = dspy.InputField(desc="The research question")
    papers: str = dspy.InputField(desc="Retrieved papers, each with its PMID")
    clinician_summary: dict = dspy.OutputField(desc="ClinicianSummary as JSON")
    technical_summary: dict = dspy.OutputField(desc="TechnicalSummary as JSON")


class DualAudienceProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateDualAudience)

    def forward(self, query, papers):
        return self.generate(query=query, papers=papers)
