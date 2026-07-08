"""Persist a versioned spectral snapshot as first-class KG nodes.

A snapshot node (IRI keyed by content hash) carries the spectral gap, node/edge
counts, timestamp, and one membership triple per entity. Versioning: the IRI's
hash changes iff the graph or its spectral summary changes, so snapshots are
immutable and comparable across retraining cycles (Scier's spectral-snapshot
requirement)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pyoxigraph as ox

_NS = "http://ex/spectral/"
_GAP0 = ox.NamedNode(_NS + "gap0")
_NNODES = ox.NamedNode(_NS + "nNodes")
_NEDGES = ox.NamedNode(_NS + "nEdges")
_AT = ox.NamedNode(_NS + "computedAt")
_MEMBER = ox.NamedNode(_NS + "community")
_XSD_INT = ox.NamedNode("http://www.w3.org/2001/XMLSchema#integer")
_XSD_DBL = ox.NamedNode("http://www.w3.org/2001/XMLSchema#double")


@dataclass
class SpectralSnapshot:
    gap: list[float]
    communities: dict[str, int]
    n_nodes: int
    n_edges: int
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def content_hash(self) -> str:
        payload = json.dumps(
            {"gap": [round(g, 6) for g in self.gap],
             "comm": sorted(self.communities.items()),
             "n": [self.n_nodes, self.n_edges]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_spectral_snapshot(
    store: ox.Store, snapshot: SpectralSnapshot, *, graph_name=None
) -> str:
    uri = f"{_NS}snapshot/{snapshot.content_hash()}"
    node = ox.NamedNode(uri)
    g = (graph_name,) if graph_name is not None else ()

    def add(p, lit):
        store.add(ox.Quad(node, p, lit, *g))

    add(_GAP0, ox.Literal(str(snapshot.gap[0]), datatype=_XSD_DBL))
    add(_NNODES, ox.Literal(str(snapshot.n_nodes), datatype=_XSD_INT))
    add(_NEDGES, ox.Literal(str(snapshot.n_edges), datatype=_XSD_INT))
    add(_AT, ox.Literal(snapshot.computed_at))
    for entity, cid in snapshot.communities.items():
        entity_iri = entity if "://" in entity else f"{_NS}entity/{entity}"
        store.add(ox.Quad(
            ox.NamedNode(entity_iri), _MEMBER,
            ox.Literal(str(cid), datatype=_XSD_INT), *g,
        ))
    return uri
