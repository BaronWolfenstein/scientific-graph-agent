"""Tests for PubMed retrieval tool — uses real NCBI API."""
import pytest
import time


def test_fetch_pubmed_results_returns_list():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("CAR-T cell lymphoma", max_results=2)
    assert isinstance(results, list)
    assert len(results) > 0
    time.sleep(0.35)


def test_pubmed_paper_has_required_fields():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("immunotherapy cancer", max_results=1)
    assert len(results) == 1
    paper = results[0]
    for key in ("id", "pmid", "title", "authors", "summary", "url", "published", "source"):
        assert key in paper, f"Missing key: {key}"
    time.sleep(0.35)


def test_pubmed_url_format():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("aspirin cardiology", max_results=1)
    assert len(results) == 1
    pmid = results[0]["pmid"]
    assert results[0]["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    time.sleep(0.35)


def test_pubmed_source_field():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("diabetes mellitus treatment", max_results=1)
    assert results[0]["source"] == "pubmed"
    time.sleep(0.35)


def test_search_pubmed_tool_is_callable():
    from agent_graph.tools import search_pubmed
    # Verify it's a LangChain tool (has .invoke)
    assert hasattr(search_pubmed, "invoke")
