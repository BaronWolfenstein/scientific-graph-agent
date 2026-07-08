"""Tests for weighted BFS traversal + context serialization."""
import pytest


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity, "snippet": "t"}


def _graph():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    kg.add_relation("CML", "disease", "subtype_of", "Leukemia", "disease", _ev("2", year=2010))
    kg.add_relation("Aspirin", "drug", "treats", "Headache", "disease", _ev("3", relevance=30))
    return kg


def test_query_finds_seed_and_neighbors():
    kg = _graph()
    edges = kg.query(["Imatinib"], max_depth=2)
    rels = {(e["subject"].split("/")[-1], e["relation"], e["object"].split("/")[-1]) for e in edges}
    assert ("imatinib", "treats", "cml") in rels
    assert ("cml", "subtype_of", "leukemia") in rels       # reached at depth 2
    assert all("aspirin" not in e["subject"] for e in edges)  # disconnected


def test_min_confidence_filter():
    kg = _graph()
    hi = kg.query(["Aspirin"], min_confidence=0.0)
    lo = kg.query(["Aspirin"], min_confidence=0.95)
    assert len(hi) >= 1 and len(lo) == 0


def test_as_of_year_filter():
    kg = _graph()
    edges = kg.query(["CML"], max_depth=1, as_of_year=2005)
    # subtype_of (2010) is excluded; treats-CML (2020) excluded; nothing <= 2005
    assert all(e["years"][0] is None or e["years"][0] <= 2005 for e in edges)


def test_hint_match_ranks_first():
    kg = _graph()
    edges = kg.query(["CML"], relation_hints=["subtype_of"], max_depth=1)
    assert edges[0]["relation"] == "subtype_of"


def test_to_context_marks_contested():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("StatinX", "drug", "increases_risk_of", "MI", "disease", _ev("1"))
    kg.add_relation("StatinX", "drug", "decreases_risk_of", "MI", "disease", _ev("2"))
    edges = kg.query(["StatinX"], max_depth=1)
    ctx = kg.to_context(edges)
    assert "CONTESTED" in ctx and "increases_risk_of" in ctx
