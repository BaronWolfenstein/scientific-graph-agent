"""Tests for the Gradio MVP's analysis layer (UI-free)."""
from agent_graph.app.analysis import analyze_store, seed_example_store


def test_seeded_store_analysis_shapes():
    kg = seed_example_store()
    res = analyze_store(kg)
    # claims present, every combined confidence in [0,1]
    assert len(res.claims) >= 5
    assert all(0.0 <= c.combined <= 1.0 for c in res.claims)
    assert all(0.0 <= c.textual <= 1.0 for c in res.claims)
    # spectral structure computed
    assert res.n_nodes > 0 and res.n_edges > 0
    assert len(set(res.communities.values())) >= 2      # example has ≥2 clusters
    assert len(res.spectral_gap) >= 2                    # λ1, λ2 at least
    assert res.top_bridges                               # some bridge ranking


def test_more_supported_claim_outranks_single_paper():
    kg = seed_example_store()
    res = analyze_store(kg)
    # entity URIs are slugified (lowercased) for URI stability
    by_key = {(c.subject, c.relation, c.object): c for c in res.claims}
    strong = by_key[("bisoprolol", "treats", "heartfailure")]   # 3 papers
    weak = by_key[("carvedilol", "treats", "cml")]              # 1 paper (the bridge)
    assert strong.support > weak.support
    assert strong.textual > weak.textual


def test_contested_pair_is_flagged():
    kg = seed_example_store()
    res = analyze_store(kg)
    contested = [c for c in res.claims if c.contested]
    assert any(c.object == "myopathy" for c in contested)


def test_focused_query_filters_claims():
    kg = seed_example_store()
    res = analyze_store(kg, ["Imatinib"])   # query slugifies the seed internally
    subjects = {c.subject for c in res.claims}
    assert "imatinib" in subjects


def test_build_demo_constructs_without_launching():
    from agent_graph.app.app import build_demo
    demo = build_demo()
    import gradio as gr
    assert isinstance(demo, gr.Blocks)
