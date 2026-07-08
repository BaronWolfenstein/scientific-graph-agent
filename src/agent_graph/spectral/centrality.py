"""Bridging centrality: betweenness weighted by the bridging coefficient, which
up-weights nodes that sit between densely-connected regions (structural holes).
bridging_coeff(v) = (1/deg(v)) / sum_{u in N(v)} 1/deg(u)."""
from __future__ import annotations

import networkx as nx


def bridging_centrality(G: nx.Graph) -> dict[str, float]:
    betw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    deg = dict(G.degree())
    out: dict[str, float] = {}
    for v in G.nodes():
        neigh = list(G.neighbors(v))
        if not neigh or deg[v] == 0:
            out[v] = 0.0
            continue
        denom = sum(1.0 / deg[u] for u in neigh if deg[u] > 0)
        coeff = (1.0 / deg[v]) / denom if denom > 0 else 0.0
        out[v] = betw[v] * coeff
    return out
