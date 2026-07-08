"""Tests for the ontology-constrained extraction schema."""
import pytest
from pydantic import ValidationError


def test_valid_triplet():
    from agent_graph.kg.extract import ScientificTriplet
    t = ScientificTriplet(subject="Imatinib", subject_type="drug",
                          relation="treats", object="CML", object_type="disease")
    assert t.polarity == "supports"


def test_off_ontology_relation_rejected():
    from agent_graph.kg.extract import ScientificTriplet
    with pytest.raises(ValidationError):
        ScientificTriplet(subject="A", subject_type="drug",
                          relation="cites", object="B", object_type="paper")


def test_off_ontology_entity_type_rejected():
    from agent_graph.kg.extract import ScientificTriplet
    with pytest.raises(ValidationError):
        ScientificTriplet(subject="A", subject_type="organization",
                          relation="treats", object="B", object_type="disease")


def test_empty_extraction_default():
    from agent_graph.kg.extract import TripletExtraction
    assert TripletExtraction().triplets == []


def test_evidence_valid_and_constrained_pmid():
    from agent_graph.kg.extract import Evidence
    e = Evidence(paper_uri="pmid:12345", pmid="12345", relevance=90)
    assert e.pmid == "12345" and e.polarity == "supports"


def test_evidence_rejects_nonnumeric_pmid_and_bad_relevance():
    from agent_graph.kg.extract import Evidence
    with pytest.raises(ValidationError):
        Evidence(paper_uri="x", pmid="PMC-not-a-pmid", relevance=50)
    with pytest.raises(ValidationError):
        Evidence(paper_uri="x", relevance=500)  # out of 0-100


def test_evidence_coerces_from_dict():
    from agent_graph.kg.extract import Evidence
    e = Evidence.model_validate({"paper_uri": "arxiv:2401.1", "relevance": 70})
    assert e.pmid is None and e.relevance == 70
