"""Beta-Bernoulli confidence for KG claims.

Each claim is a Beta(alpha, beta) belief. Supporting papers add to alpha,
refuting papers add to beta, weighted by relevance as fractional pseudo-counts.
Point estimate = posterior mean; ranking uses a credible-interval lower bound
so thin evidence is penalized.
"""
from math import sqrt

PRIOR_A = 1.0
PRIOR_B = 1.0


def evidence_weight(relevance: float) -> float:
    return 0.3 + 0.65 * (relevance / 100.0)


def beta_params(evidence: list, prior_a: float = PRIOR_A, prior_b: float = PRIOR_B):
    a = prior_a + sum(
        evidence_weight(e.get("relevance", 50))
        for e in evidence if e.get("polarity", "supports") == "supports"
    )
    b = prior_b + sum(
        evidence_weight(e.get("relevance", 50))
        for e in evidence if e.get("polarity") == "refutes"
    )
    return a, b


def confidence(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def confidence_lb(alpha: float, beta: float, z: float = 1.64) -> float:
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return max(0.0, mean - z * sqrt(var))
