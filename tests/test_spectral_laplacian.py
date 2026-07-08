import numpy as np, networkx as nx
from agent_graph.spectral.laplacian import (
    combinatorial_laplacian, spectral_embedding, spectral_gap)

def _two_triangles():
    G = nx.Graph()
    for a, b in [("a","b"),("b","c"),("a","c"),("x","y"),("y","z"),("x","z")]:
        G.add_edge(a, b, weight=1.0)
    return G

def test_laplacian_is_psd_and_row_sums_zero():
    G = _two_triangles()
    nodes, L = combinatorial_laplacian(G)
    assert L.shape == (6, 6)
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0)  # L = D - A

def test_fiedler_separates_components():
    G = _two_triangles()
    eigs = spectral_gap(G, k=3)
    assert eigs[0] < 1e-8 and eigs[1] < 1e-8            # 2 zero eigenvalues = 2 comps
    nodes, emb = spectral_embedding(G, k=2)
    tri1 = np.mean([emb[nodes.index(n), 0] for n in ("a","b","c")])
    tri2 = np.mean([emb[nodes.index(n), 0] for n in ("x","y","z")])
    assert abs(tri1 - tri2) > 1e-6
