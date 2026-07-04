"""Tests for the pre-HITL JSON Schema + citation-grounding validation gate."""
import pytest


def _valid_clinician(pmid="111"):
    return {"audience": "clinician", "bottom_line": "Drug X works.",
            "key_findings": ["finding 1"],
            "evidence": [{"claim": "c", "pmid": pmid, "source_url": "u"}],
            "confidence_note": "well supported"}


def _valid_technical(pmid="111"):
    return {"audience": "technical", "detailed_findings": "d",
            "methodology_notes": "m",
            "evidence": [{"claim": "c", "pmid": pmid, "source_url": "u"}],
            "caveats": ["caveat 1"]}


def _papers(pmid="111"):
    return [{"id": pmid, "pmid": pmid, "title": "T", "relevance_score": 90}]


def test_valid_output_has_no_errors_and_routes_approve():
    from agent_graph.nodes import validate_output_node, route_after_validation
    state = {"clinician_summary": _valid_clinician(), "technical_summary": _valid_technical(),
             "papers": _papers(), "iteration": 0, "max_iterations": 2}
    out = validate_output_node(state)
    assert out["validation_errors"] == []
    assert route_after_validation({**state, **out}) == "approve"


def test_structural_violation_flagged_and_routes_regenerate():
    from agent_graph.nodes import validate_output_node, route_after_validation
    bad = _valid_clinician()
    del bad["bottom_line"]  # required field missing -> JSON Schema violation
    state = {"clinician_summary": bad, "technical_summary": _valid_technical(),
             "papers": _papers(), "iteration": 0, "max_iterations": 2}
    out = validate_output_node(state)
    assert any("clinician_summary" in e for e in out["validation_errors"])
    assert route_after_validation({**state, **out}) == "regenerate"


def test_ungrounded_citation_flagged():
    from agent_graph.nodes import validate_output_node
    state = {"clinician_summary": _valid_clinician(pmid="999"),  # 999 not retrieved
             "technical_summary": _valid_technical(pmid="111"),
             "papers": _papers(pmid="111"), "iteration": 0, "max_iterations": 2}
    out = validate_output_node(state)
    assert any("999" in e for e in out["validation_errors"])


def test_routing_breaks_loop_at_max_iterations():
    from agent_graph.nodes import route_after_validation
    state = {"validation_errors": ["x"], "iteration": 2, "max_iterations": 2}
    assert route_after_validation(state) == "approve"  # give up, never loop forever


def test_demo_graph_includes_validate_output():
    from agent_graph.graph import create_demo_graph
    g = create_demo_graph()
    assert "validate_output" in g.get_graph().nodes


def test_pmid_collision_conflicting_sources_flagged():
    """One PMID cited for two different papers (conflicting sources) — the
    LIGER/Seurat-style collision — must be flagged."""
    from agent_graph.nodes import validate_output_node
    cs = {"audience": "clinician", "bottom_line": "b", "key_findings": ["f"],
          "evidence": [{"claim": "seurat", "pmid": "31178118",
                        "source_url": "https://doi.org/10.1016/j.cell.2019.05.031"}],
          "confidence_note": "n"}
    ts = {"audience": "technical", "detailed_findings": "d", "methodology_notes": "m",
          "evidence": [{"claim": "liger", "pmid": "31178118",
                        "source_url": "https://doi.org/10.1016/j.cell.2019.05.006"}],
          "caveats": ["cav"]}
    state = {"clinician_summary": cs, "technical_summary": ts,
             "papers": [{"id": "31178118", "pmid": "31178118", "title": "T"}],
             "iteration": 0, "max_iterations": 2}
    out = validate_output_node(state)
    assert any("31178118" in e and "conflicting" in e for e in out["validation_errors"])


def test_pmid_source_mismatch_vs_retrieval_flagged():
    """A cited PMID whose source_url disagrees with the retrieved record
    (mis-attribution) must be flagged — using retrieval data, no extra API."""
    from agent_graph.nodes import validate_output_node
    ev = [{"claim": "c", "pmid": "111", "source_url": "https://doi.org/wrong"}]
    cs = {"audience": "clinician", "bottom_line": "b", "key_findings": ["f"],
          "evidence": ev, "confidence_note": "n"}
    ts = {"audience": "technical", "detailed_findings": "d", "methodology_notes": "m",
          "evidence": ev, "caveats": ["cav"]}
    state = {"clinician_summary": cs, "technical_summary": ts,
             "papers": [{"id": "111", "pmid": "111", "title": "T",
                         "url": "https://pubmed.ncbi.nlm.nih.gov/111/"}],
             "iteration": 0, "max_iterations": 2}
    out = validate_output_node(state)
    assert any("111" in e and "retrieved record" in e for e in out["validation_errors"])
