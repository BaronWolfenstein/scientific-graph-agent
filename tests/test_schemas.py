"""Tests for Pydantic schemas and the JSON extraction + retry helpers."""
import json
import pytest
from pydantic import ValidationError


def test_evidence_model_valid():
    from agent_graph.schemas import Evidence
    e = Evidence(claim="Drug X reduces mortality", pmid="12345678",
                 source_url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    assert e.pmid == "12345678"


def test_clinician_summary_valid():
    from agent_graph.schemas import ClinicianSummary, Evidence
    cs = ClinicianSummary(
        bottom_line="CAR-T shows durable remission in DLBCL.",
        key_findings=["60% CR rate at 12 months"],
        evidence=[Evidence(claim="CR rate", pmid="34567890",
                           source_url="https://pubmed.ncbi.nlm.nih.gov/34567890/")],
        confidence_note="Based on 2 RCTs."
    )
    assert cs.audience == "clinician"


def test_technical_summary_valid():
    from agent_graph.schemas import TechnicalSummary, Evidence
    ts = TechnicalSummary(
        detailed_findings="CD19-directed CAR-T achieved 58% ORR.",
        methodology_notes="Phase II, n=93, LBCL >= 2 prior lines.",
        evidence=[Evidence(claim="ORR", pmid="34567890",
                           source_url="https://pubmed.ncbi.nlm.nih.gov/34567890/")],
        caveats=["Short follow-up", "Single arm"]
    )
    assert ts.audience == "technical"


def test_clinician_summary_missing_required_field_raises():
    from agent_graph.schemas import ClinicianSummary
    with pytest.raises(ValidationError):
        ClinicianSummary(key_findings=[], evidence=[], confidence_note="ok")
        # missing bottom_line


def test_extract_json_strips_markdown_fence():
    from agent_graph.nodes import _extract_json
    raw = '```json\n{"key": "value"}\n```'
    assert _extract_json(raw) == '{"key": "value"}'


def test_extract_json_passthrough_plain():
    from agent_graph.nodes import _extract_json
    raw = '{"key": "value"}'
    assert _extract_json(raw) == '{"key": "value"}'


def test_generate_with_retry_succeeds_first_attempt(monkeypatch):
    from agent_graph.nodes import _generate_with_retry
    from agent_graph.schemas import ClinicianSummary, Evidence
    from langchain_core.messages import SystemMessage

    valid_json = json.dumps({
        "bottom_line": "Drug reduces mortality.",
        "key_findings": ["50% reduction"],
        "evidence": [{"claim": "reduction", "pmid": "111",
                      "source_url": "https://pubmed.ncbi.nlm.nih.gov/111/"}],
        "confidence_note": "Single RCT."
    })

    class FakeLLM:
        def invoke(self, messages):
            class Resp:
                content = valid_json
            return Resp()

    result, first_err = _generate_with_retry(FakeLLM(), [SystemMessage(content="test")], ClinicianSummary)
    assert isinstance(result, ClinicianSummary)
    assert first_err is None


def test_generate_with_retry_retries_on_bad_json(monkeypatch):
    from agent_graph.nodes import _generate_with_retry
    from agent_graph.schemas import ClinicianSummary
    from langchain_core.messages import SystemMessage

    valid_json = json.dumps({
        "bottom_line": "Drug reduces mortality.",
        "key_findings": ["50% reduction"],
        "evidence": [{"claim": "reduction", "pmid": "111",
                      "source_url": "https://pubmed.ncbi.nlm.nih.gov/111/"}],
        "confidence_note": "Single RCT."
    })

    call_count = {"n": 0}

    class FakeLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            class Resp:
                content = "not valid json" if call_count["n"] == 1 else valid_json
            return Resp()

    result, first_err = _generate_with_retry(FakeLLM(), [SystemMessage(content="test")], ClinicianSummary)
    assert isinstance(result, ClinicianSummary)
    assert first_err is not None  # first attempt failed
    assert call_count["n"] == 2   # retried exactly once
