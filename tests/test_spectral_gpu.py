"""GPU (CuPy) Laplacian-eig backend parity. Skips when cupy is absent (CPU-only
CI); on a GPU box the gpu backend must reproduce the scipy/numpy CPU path to
tolerance. Eigenvalues are sign-free; the embedding is compared by its pairwise
distance matrix, which is invariant to the per-eigenvector sign ambiguity."""
import numpy as np
import pytest

pytest.importorskip("cupy")     # skips off-box; runs on a CuPy/GPU box

import networkx as nx  # noqa: E402
from agent_graph.spectral.laplacian import spectral_gap, spectral_embedding  # noqa: E402


def _graph():
    G = nx.Graph()
    for a, b in [("a", "b"), ("b", "c"), ("a", "c"),
                 ("x", "y"), ("y", "z"), ("x", "z"),
                 ("b", "x")]:
        G.add_edge(a, b, weight=1.0)
    return G


def test_gpu_spectral_gap_matches_cpu():
    G = _graph()
    cpu = np.asarray(spectral_gap(G, k=4, backend="cpu"))
    gpu = np.asarray(spectral_gap(G, k=4, backend="gpu"))
    assert np.allclose(cpu, gpu, atol=1e-6)


def test_gpu_spectral_embedding_matches_cpu_up_to_sign():
    G = _graph()
    _, cpu = spectral_embedding(G, k=3, backend="cpu")
    _, gpu = spectral_embedding(G, k=3, backend="gpu")

    def pdist(C):                       # sign-invariant summary of the embedding
        d = C[:, None, :] - C[None, :, :]
        return np.sqrt((d ** 2).sum(axis=2))

    assert np.allclose(pdist(cpu), pdist(gpu), atol=1e-6)


def test_gpu_louvain_recovers_planted_communities():
    pytest.importorskip("cugraph")
    from agent_graph.spectral.communities import detect_communities
    G = _graph()   # two triangles joined by a single weak bridge (b-x)
    part = detect_communities(G, method="louvain", backend="gpu")
    # cuGraph Louvain won't match networkx labels, but must recover the STRUCTURE:
    assert part["a"] == part["b"] == part["c"]      # triangle 1 together
    assert part["x"] == part["y"] == part["z"]      # triangle 2 together
    assert part["a"] != part["x"]                   # the two triangles split


def test_gpu_leiden_recovers_planted_communities():
    pytest.importorskip("cugraph")
    from agent_graph.spectral.communities import detect_communities
    G = _graph()
    part = detect_communities(G, method="leiden", backend="gpu")
    assert part["a"] == part["b"] == part["c"]
    assert part["x"] == part["y"] == part["z"]
    assert part["a"] != part["x"]
