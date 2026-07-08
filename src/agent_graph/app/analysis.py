"""Analysis layer for the Gradio MVP — pure, UI-free, and unit-testable.

Ties the three portfolio pieces together over one knowledge graph:
  - the KG's textual (Beta-Bernoulli) confidence per claim edge (`query`),
  - the spectral layer's structure (communities, spectral gap, bridging centrality),
  - the confidence toolkit's combine rule (textual + structural legs; the
    empirical leg stays empty in SGA per the data-linkage gap).

`seed_example_store` builds a small demo KG so the app runs offline (no live
retrieval / no API key). The Gradio layer in `app.py` is a thin wrapper over
`analyze_store`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent_graph.kg import get_knowledge_graph
from agent_graph.kg.confidence_toolkit import combine_confidence
from agent_graph.spectral import (
    bridging_centrality,
    build_entity_graph,
    detect_communities,
    spectral_gap,
)


@dataclass
class ClaimAnalysis:
    subject: str
    relation: str
    object: str
    textual: float          # Beta-Bernoulli confidence
    structural: float       # graph-structure signal
    combined: float         # confidence-toolkit combine
    support: int            # number of papers
    contested: bool


@dataclass
class AnalysisResult:
    claims: List[ClaimAnalysis]
    communities: dict          # {entity_uri: community_id}
    spectral_gap: List[float]
    top_bridges: List[Tuple[str, float]]
    n_nodes: int
    n_edges: int


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity,
            "snippet": ""}


def seed_example_store():
    """A small demo KG: two topical clusters bridged by one cross-cluster claim,
    with varied support (confidence) and one contested pair — enough structure to
    show communities, a bridge, and confidence differences."""
    kg = get_knowledge_graph()
    # Cluster 1 — cardiology (built up with multiple papers -> higher confidence)
    for pmid in ("101", "102", "103"):
        kg.add_relation("Bisoprolol", "drug", "treats", "HeartFailure", "disease", _ev(pmid))
    kg.add_relation("Carvedilol", "drug", "treats", "HeartFailure", "disease", _ev("104"))
    kg.add_relation("Bisoprolol", "drug", "decreases_risk_of", "Arrhythmia", "disease", _ev("105"))
    # Cluster 2 — oncology
    for pmid in ("201", "202"):
        kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev(pmid))
    kg.add_relation("Dasatinib", "drug", "treats", "CML", "disease", _ev("203"))
    # Bridge — a cardiology drug linked to an oncology disease (single paper)
    kg.add_relation("Carvedilol", "drug", "treats", "CML", "disease", _ev("300"))
    # Contested pair — mutually exclusive claims flag both contested
    kg.add_relation("StatinX", "drug", "increases_risk_of", "Myopathy", "disease", _ev("400"))
    kg.add_relation("StatinX", "drug", "decreases_risk_of", "Myopathy", "disease", _ev("401"))
    return kg


def _structural_confidence(subj: str, obj: str, communities: dict) -> float:
    """Intra-community edges are structurally consolidated (endpoints cluster
    together); cross-community (bridge) edges are structurally surprising."""
    if subj in communities and obj in communities:
        return 1.0 if communities[subj] == communities[obj] else 0.5
    return 0.5


def analyze_store(kg, entities: Optional[List[str]] = None, *,
                  min_confidence: float = 0.0, max_depth: int = 3) -> AnalysisResult:
    """Run the full analysis over the KG's current claims."""
    G = build_entity_graph(kg.store)
    n = G.number_of_nodes()
    communities = detect_communities(G) if n else {}
    gap = spectral_gap(G, k=min(6, max(1, n - 1))) if G.number_of_edges() else []
    bridges = bridging_centrality(G) if n else {}
    top_bridges = sorted(bridges.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # entities to seed the claim query: user-supplied tokens, else every node's
    # human-readable local name (so "analyze everything" works out of the box).
    if not entities:
        entities = [uri.rstrip("/").split("/")[-1] for uri in G.nodes()]

    edges = kg.query(entities, min_confidence=min_confidence, max_depth=max_depth)
    claims = []
    for e in edges:
        struct = _structural_confidence(e["subject"], e["object"], communities)
        claims.append(ClaimAnalysis(
            subject=e["subject"].rstrip("/").split("/")[-1],
            relation=e["relation"],
            object=e["object"].rstrip("/").split("/")[-1],
            textual=round(e["confidence"], 3),
            structural=struct,
            combined=round(combine_confidence(e["confidence"], structural=struct), 3),
            support=e["support"],
            contested=e["contested"],
        ))
    claims.sort(key=lambda c: c.combined, reverse=True)
    return AnalysisResult(
        claims=claims, communities=communities, spectral_gap=gap,
        top_bridges=top_bridges, n_nodes=n, n_edges=G.number_of_edges(),
    )
