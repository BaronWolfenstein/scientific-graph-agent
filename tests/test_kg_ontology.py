"""Tests for the frozen KG ontology and URI minting."""
import pytest


def test_slugify_normalizes():
    from agent_graph.kg.ontology import slugify
    assert slugify("  Imatinib Mesylate ") == "imatinib-mesylate"
    assert slugify("Non-Small_Cell") == "non-small-cell"
    assert slugify("CD19+") == "cd19"


def test_entity_uri():
    from agent_graph.kg.ontology import entity_uri
    assert entity_uri("drug", "Imatinib") == "https://kg.local/drug/imatinib"


def test_paper_uri_precedence():
    from agent_graph.kg.ontology import paper_uri
    assert paper_uri(pmid="12345") == "pmid:12345"
    assert paper_uri(arxiv_id="2401.001") == "arxiv:2401.001"
    assert paper_uri(paper_id="Some Local ID") == "paper:some-local-id"


def test_mutex_is_symmetric_and_partners():
    from agent_graph.kg.ontology import mutex_partners
    assert "decreases_risk_of" in mutex_partners("increases_risk_of")
    assert "increases_risk_of" in mutex_partners("decreases_risk_of")
    assert mutex_partners("treats") == []


def test_functional_is_empty_but_present():
    from agent_graph.kg import ontology
    assert ontology.FUNCTIONAL == set()


def test_ontology_frozen_membership():
    from agent_graph.kg.ontology import ENTITY_TYPES, RELATION_TYPES
    assert set(ENTITY_TYPES) == {
        "drug", "disease", "gene", "biomarker", "method", "population", "paper", "author"
    }
    for rel in ("treats", "increases_risk_of", "subtype_of", "wrote"):
        assert rel in RELATION_TYPES
    assert "affiliated_with" not in RELATION_TYPES  # reserved
    assert "cites" not in RELATION_TYPES            # reserved
