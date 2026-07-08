"""Round-trip serialization and branch merge."""
import pytest


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity, "snippet": "t"}


def test_to_dict_from_dict_roundtrip():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    blob = kg.to_dict()
    kg2 = OxigraphKG.from_dict(blob)
    e1 = kg.query(["Imatinib"]); e2 = kg2.query(["Imatinib"])
    assert len(e1) == len(e2) == 1
    assert abs(e1[0]["confidence"] - e2[0]["confidence"]) < 1e-9


def test_merge_combines_evidence_without_double_counting():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    from agent_graph.kg import merge_graphs
    a = OxigraphKG(); a.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    b = OxigraphKG(); b.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("2"))
    b.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))  # overlap
    merged = merge_graphs(a, b)
    meta = merged._claim_meta("drug", "Imatinib", "treats", "disease", "CML")
    assert meta["support"] == 2  # papers 1 and 2, not 3


def test_merge_graphs_handles_none():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    from agent_graph.kg import merge_graphs
    a = OxigraphKG(); a.add_relation("X", "drug", "treats", "Y", "disease", _ev("1"))
    assert merge_graphs(None, a) is a
    assert merge_graphs(a, None) is a


def test_factory_returns_protocol():
    from agent_graph.kg import get_knowledge_graph
    from agent_graph.kg.protocol import KnowledgeGraph
    kg = get_knowledge_graph()
    assert isinstance(kg, KnowledgeGraph)
