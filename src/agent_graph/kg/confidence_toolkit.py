"""KG confidence toolkit — combine multiple confidence sources for a claim edge.

Three legs, each a score in [0, 1]:
  - **textual**    Beta-Bernoulli cross-paper evidence (`confidence.py`). Always
                   present — it's how many independent papers assert the edge.
  - **structural** how the edge sits in the graph (the `spectral` layer:
                   community coherence / bridging position). Optional — present
                   once the spectral snapshot has run.
  - **empirical**  data-driven conditional-independence verdict from a zero-flow
                   CI test (the *estimator* lives in causal_bench #85, not here).
                   Sparse — present only when the edge is linked to a dataset.
                   Represented here as a reifier-ready verdict SLOT so the KG is
                   complete-by-design; the SGA claim graph carries no per-edge
                   data, so this leg stays empty until a data-linkage layer exists.

The combine rule is a weighted average over the *present* legs (weights
renormalized), so it degrades gracefully to textual-only. An "underpowered"
empirical verdict is non-informative and is dropped, not scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

EmpiricalVerdict = Literal["supports", "refutes", "underpowered"]

# verdict -> confidence contribution in [0, 1]; "underpowered" -> None (dropped:
# the CI test could not decide, so it must not move the combined score).
_VERDICT_SCORE = {"supports": 0.9, "refutes": 0.1}

DEFAULT_WEIGHTS = {"textual": 0.5, "structural": 0.25, "empirical": 0.25}


@dataclass(frozen=True)
class EmpiricalCIResult:
    """Reifier-ready empirical verdict from a zero-flow CI test. The estimator
    itself is causal_bench #85 (rectified-flow CI, needs data + torch); this is
    only the slot SGA attaches to a claim edge when a dataset is linked."""

    verdict: EmpiricalVerdict
    test: str                    # e.g. "zero-flow-ci"
    effective_n: int

    def score(self) -> Optional[float]:
        """Confidence contribution in [0, 1], or None if non-informative."""
        return _VERDICT_SCORE.get(self.verdict)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def combine_confidence(
    textual: float,
    *,
    structural: Optional[float] = None,
    empirical: Optional[EmpiricalCIResult] = None,
    weights: Optional[dict] = None,
) -> float:
    """Combine the confidence legs into one score in [0, 1].

    Weighted average over the legs that are present, with the weights of the
    present legs renormalized to sum to 1. `textual` is always present; a `None`
    structural leg or a missing/underpowered empirical leg is simply omitted, so
    the rule reduces to the textual score when nothing else is known.
    """
    w = weights or DEFAULT_WEIGHTS
    legs = [("textual", _clamp01(textual))]
    if structural is not None:
        legs.append(("structural", _clamp01(structural)))
    if empirical is not None:
        es = empirical.score()                 # None when underpowered -> dropped
        if es is not None:
            legs.append(("empirical", es))
    total_w = sum(w[name] for name, _ in legs)
    if total_w <= 0:
        return _clamp01(textual)
    return sum(w[name] * score for name, score in legs) / total_w
