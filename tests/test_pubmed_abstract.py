"""Regression tests for PubMed abstract parsing.

A PubMed article can have an empty <AbstractText/>, a structured multi-section
abstract, or nested markup. Parsing must always yield a string (never None), or
downstream `paper['summary'][:N]` crashes (found by the GEPA harvest run).
"""
import xml.etree.ElementTree as ET
import pytest


def _article(inner_xml):
    return ET.fromstring(f"<PubmedArticle>{inner_xml}</PubmedArticle>")


def test_empty_abstract_returns_empty_string_not_none():
    from agent_graph.tools import _abstract_text
    art = _article("<Abstract><AbstractText/></Abstract>")
    assert _abstract_text(art) == ""


def test_missing_abstract_returns_empty_string():
    from agent_graph.tools import _abstract_text
    assert _abstract_text(_article("")) == ""


def test_structured_abstract_joins_sections():
    from agent_graph.tools import _abstract_text
    art = _article("<Abstract>"
                   "<AbstractText Label='BACKGROUND'>bg text</AbstractText>"
                   "<AbstractText Label='METHODS'>method text</AbstractText>"
                   "</Abstract>")
    out = _abstract_text(art)
    assert "bg text" in out and "method text" in out


def test_nested_markup_in_abstract_is_flattened():
    from agent_graph.tools import _abstract_text
    art = _article("<Abstract><AbstractText>alpha <i>beta</i> gamma</AbstractText></Abstract>")
    assert _abstract_text(art) == "alpha beta gamma"
