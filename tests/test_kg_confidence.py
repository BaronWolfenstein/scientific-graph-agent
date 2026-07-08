"""Tests for Beta-Bernoulli confidence accumulation."""
import pytest


def test_more_support_raises_confidence():
    from agent_graph.kg.confidence import beta_params, confidence
    one = [{"relevance": 80, "polarity": "supports"}]
    three = one * 3
    assert confidence(*beta_params(three)) > confidence(*beta_params(one))


def test_refutation_lowers_confidence():
    from agent_graph.kg.confidence import beta_params, confidence
    supp = [{"relevance": 80, "polarity": "supports"}]
    mixed = supp + [{"relevance": 80, "polarity": "refutes"}] * 3
    assert confidence(*beta_params(mixed)) < confidence(*beta_params(supp))


def test_thin_evidence_penalized_by_lower_bound():
    from agent_graph.kg.confidence import beta_params, confidence, confidence_lb
    thin = [{"relevance": 90, "polarity": "supports"}]
    thick = [{"relevance": 60, "polarity": "supports"}] * 8
    # thick may have similar/lower mean but a higher lower-bound (less uncertainty)
    assert confidence_lb(*beta_params(thick)) > confidence_lb(*beta_params(thin))


def test_defaults_for_missing_fields():
    from agent_graph.kg.confidence import beta_params
    a, b = beta_params([{}])  # relevance->50, polarity->supports
    assert a > 1.0 and b == 1.0
