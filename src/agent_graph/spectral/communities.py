"""Community detection on the entity graph. Louvain (networkx built-in, no
extra deps) is the default; Leiden is available via the `leiden` extra and is
lazy-imported so CPU-only installs don't require igraph/leidenalg.

An optional GPU backend (`backend="gpu"`, or `"auto"`) runs cuGraph Louvain via
`spectral.gpu` (gated behind the `[gpu]` extra). Note: cuGraph Louvain is a
different algorithm instance than networkx Louvain, so it recovers the same
community *structure* on well-separated graphs, not identical labels."""
from __future__ import annotations

import networkx as nx

from .gpu import resolve_backend


def detect_communities(
    G: nx.Graph, method: str = "louvain", resolution: float = 1.0, seed: int = 0,
    backend: str = "cpu",
) -> dict[str, int]:
    if resolve_backend(backend) == "gpu":
        from .gpu import louvain_gpu, leiden_gpu
        if method == "louvain":
            return louvain_gpu(G, resolution=resolution)
        if method == "leiden":
            return leiden_gpu(G, resolution=resolution)
        raise ValueError(f"unknown method: {method!r} (use 'louvain' or 'leiden')")
    if method == "louvain":
        comms = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=seed
        )
    elif method == "leiden":
        comms = _leiden(G, resolution=resolution, seed=seed)
    else:
        raise ValueError(f"unknown method: {method!r} (use 'louvain' or 'leiden')")
    return {node: cid for cid, community in enumerate(comms) for node in community}


def _leiden(G: nx.Graph, resolution: float, seed: int):
    import igraph as ig  # lazy: only needed for method="leiden"
    import leidenalg as la

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges)
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )
    return [frozenset(nodes[i] for i in group) for group in part]
