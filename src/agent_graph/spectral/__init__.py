"""Spectral graph layer over the RDF-star knowledge graph (additive).

The structural-confidence leg of the KG confidence toolkit: Laplacian spectrum,
community detection, and centrality over the entity graph, plus versioned
spectral snapshots persisted as first-class KG nodes. Exports are filled in as
the module tasks (A2–A6) land.
"""
from .graph_adapter import build_entity_graph
from .laplacian import combinatorial_laplacian, spectral_embedding, spectral_gap
from .communities import detect_communities
from .centrality import bridging_centrality
from .snapshots import SpectralSnapshot, write_spectral_snapshot
from .transport import (
    sinkhorn_divergence, heat_kernel_cost, gromov_wasserstein, graph_gw_distance,
)
from .gnn import normalized_adjacency, sgc_propagate, train_sgc, sgc_predict

__all__ = [
    "build_entity_graph", "combinatorial_laplacian", "spectral_embedding",
    "spectral_gap", "detect_communities", "bridging_centrality",
    "SpectralSnapshot", "write_spectral_snapshot",
    "sinkhorn_divergence", "heat_kernel_cost", "gromov_wasserstein",
    "graph_gw_distance",
    "normalized_adjacency", "sgc_propagate", "train_sgc", "sgc_predict",
]
