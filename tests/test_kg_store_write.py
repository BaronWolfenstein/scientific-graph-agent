"""Tests for OxigraphKG writes: confidence accrual + contradiction flagging."""
import pytest


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity,
            "snippet": "t"}


def test_add_relation_creates_claim_with_confidence():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    meta = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")
    assert meta is not None
    assert 0.0 < meta["confidence"] <= 1.0
    assert meta["support"] == 1


def test_more_papers_raise_confidence_and_are_idempotent():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    c1 = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")["confidence"]
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("2"))
    c2 = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")["confidence"]
    assert c2 > c1
    # re-adding paper "2" must NOT increase support (idempotent per paper)
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("2"))
    meta = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")
    assert meta["support"] == 2


def test_mutex_flags_both_claims_contested():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("StatinX", "drug", "increases_risk_of", "MI", "disease", _ev("1"))
    note = kg.add_relation("StatinX", "drug", "decreases_risk_of", "MI", "disease", _ev("2"))
    assert note is not None and "contradiction" in note
    m_inc = kg._claim_meta("drug", "StatinX", "increases_risk_of", "disease", "MI")
    m_dec = kg._claim_meta("drug", "StatinX", "decreases_risk_of", "disease", "MI")
    assert m_inc["contested"] is True and m_dec["contested"] is True


def test_add_relation_rejects_malformed_evidence_at_boundary():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    from pydantic import ValidationError
    kg = OxigraphKG()
    with pytest.raises(ValidationError):
        kg.add_relation("Imatinib", "drug", "treats", "CML", "disease",
                        {"paper_uri": "x", "pmid": "not-numeric", "relevance": 50})
    # nothing was written — the blob never reached the store
    assert kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML") is None
