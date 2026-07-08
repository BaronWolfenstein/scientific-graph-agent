"""Build an undirected NetworkX graph of entities from the KG Store.

Entities are triple subjects/objects; claim predicates are edges. Edge weight
is the accumulated Beta-Bernoulli confidence on the claim's reifier when
available (the KG plan stores confidence as a reifier annotation), else 1.0.
Undirected + combinatorial because the Laplacian spectrum is defined on the
symmetric adjacency; direction is preserved in the KG itself, not here.
"""
from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx
import pyoxigraph as ox

# Predicates that are structural (rdf:type, reifies, provenance) and must NOT
# become graph edges. Extend as the KG ontology grows.
_NON_EDGE_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies",
}


def build_entity_graph(
    store: ox.Store,
    *,
    predicates: Optional[Iterable[str]] = None,
    weight_by_confidence: bool = True,
) -> nx.Graph:
    allow = set(predicates) if predicates is not None else None
    G = nx.Graph()
    for quad in store.quads_for_pattern(None, None, None, None):
        p = quad.predicate.value
        if p in _NON_EDGE_PREDICATES:
            continue
        if allow is not None and p not in allow:
            continue
        s, o = quad.subject, quad.object
        # entity→entity edges only (both endpoints are NamedNodes)
        if not (isinstance(s, ox.NamedNode) and isinstance(o, ox.NamedNode)):
            continue
        w = 1.0
        if G.has_edge(s.value, o.value):
            w += G[s.value][o.value].get("weight", 1.0)
        G.add_edge(s.value, o.value, weight=w, predicate=p)
    return G
