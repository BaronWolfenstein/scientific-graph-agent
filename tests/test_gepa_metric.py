"""Tests for the GEPA summarizer metric — deterministic gates; judges stubbed."""
import pytest


def _clin(pmid="111"):
    return {"audience": "clinician", "bottom_line": "Drug X works.",
            "key_findings": ["finding"],
            "evidence": [{"claim": "c", "pmid": pmid, "source_url": "u"}],
            "confidence_note": "well supported"}


def _tech(pmid="111"):
    return {"audience": "technical", "detailed_findings": "d", "methodology_notes": "m",
            "evidence": [{"claim": "c", "pmid": pmid, "source_url": "u"}],
            "caveats": ["caveat"]}


def _papers(pmid="111"):
    return [{"id": pmid, "pmid": pmid, "title": "T", "summary": "abstract"}]


class _Gold:
    def __init__(self, query, papers):
        self.query, self.papers = query, papers


class _Pred:
    def __init__(self, cs, ts):
        self.clinician_summary, self.technical_summary = cs, ts


_HIGH_FAITH = lambda text, papers: (1.0, 3, 3)
_HIGH_REL = lambda q, text: (1.0, "fully answers")
_LOW_REL = lambda q, text: (0.0, "does not answer the query")


def test_schema_invalid_scores_zero_with_feedback():
    from agent_graph.optimize.metric import summarizer_metric
    bad = _clin()
    del bad["bottom_line"]  # required field missing
    out = summarizer_metric(_Gold("q", _papers()), _Pred(bad, _tech()),
                            faithfulness_fn=_HIGH_FAITH, relevance_fn=_HIGH_REL)
    assert out.score == 0.0
    assert "bottom_line" in out.feedback


def test_ungrounded_citation_penalized_and_named():
    from agent_graph.optimize.metric import summarizer_metric
    out = summarizer_metric(_Gold("q", _papers(pmid="111")),
                            _Pred(_clin(pmid="999"), _tech(pmid="111")),
                            faithfulness_fn=_HIGH_FAITH, relevance_fn=_HIGH_REL)
    assert out.score < 0.5           # heavy grounding penalty despite high judges
    assert "999" in out.feedback


def test_clean_high_quality_scores_high():
    from agent_graph.optimize.metric import summarizer_metric
    out = summarizer_metric(_Gold("q", _papers()), _Pred(_clin(), _tech()),
                            faithfulness_fn=_HIGH_FAITH, relevance_fn=_HIGH_REL)
    assert out.score > 0.9


def test_low_relevance_defeats_degenerate_summary():
    from agent_graph.optimize.metric import summarizer_metric
    # faithful but off-topic (the degenerate case) must NOT score high
    out = summarizer_metric(_Gold("q", _papers()), _Pred(_clin(), _tech()),
                            faithfulness_fn=_HIGH_FAITH, relevance_fn=_LOW_REL)
    assert out.score < 0.6
    assert "does not answer" in out.feedback
