"""Tests for graph_builder_node with a stubbed LLM."""
import pytest


class _FakeStructured:
    def __init__(self, result): self._result = result
    def invoke(self, _messages): return self._result


class _FakeLLM:
    def __init__(self, result=None, raises=False):
        self._result, self._raises = result, raises
    def with_structured_output(self, _schema):
        if self._raises:
            class _Boom:
                def invoke(self, _m): raise RuntimeError("extraction failed")
            return _Boom()
        return _FakeStructured(self._result)


def _paper():
    return {"id": "P1", "pmid": "111", "title": "Imatinib in CML",
            "authors": ["Alice Smith", "Bob Jones"], "published": "2020-05-01",
            "summary": "Imatinib treats CML.", "url": "u", "relevance_score": 90}


def test_builder_adds_claims_and_wrote_edges(monkeypatch):
    import agent_graph.nodes as nodes
    from agent_graph.kg.extract import TripletExtraction, ScientificTriplet
    result = TripletExtraction(triplets=[
        ScientificTriplet(subject="Imatinib", subject_type="drug",
                          relation="treats", object="CML", object_type="disease")
    ])
    monkeypatch.setattr(nodes, "get_llm", lambda **kw: _FakeLLM(result=result))
    out = nodes.graph_builder_node({"papers": [_paper()]})
    kg = out["knowledge_graph"]
    assert kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")["support"] == 1
    # free authorship edge: Alice wrote pmid:111
    edges = kg.query(["Alice Smith"])
    assert any(e["relation"] == "wrote" for e in edges)


def test_builder_skips_papers_on_extraction_failure(monkeypatch):
    import agent_graph.nodes as nodes
    monkeypatch.setattr(nodes, "get_llm", lambda **kw: _FakeLLM(raises=True))
    out = nodes.graph_builder_node({"papers": [_paper()]})
    # pipeline not blocked; authorship edges still added from metadata
    kg = out["knowledge_graph"]
    assert kg.query(["Alice Smith"])  # wrote edge present despite extraction failure
