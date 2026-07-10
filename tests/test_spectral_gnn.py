"""Spectral GNN — Simplified Graph Convolution (SGC). Renormalized-adjacency
feature propagation Â^k X + a softmax classifier; a spectral node-classification
baseline over the entity graph. CPU/numpy, fully testable."""
import numpy as np
import networkx as nx
import pytest

from agent_graph.spectral.gnn import (
    normalized_adjacency, sgc_propagate, train_sgc, sgc_predict,
)


def _bridged_triangles():
    G = nx.Graph()
    for a, b in [("a", "b"), ("b", "c"), ("a", "c"),
                 ("x", "y"), ("y", "z"), ("x", "z"),
                 ("b", "x")]:
        G.add_edge(a, b, weight=1.0)
    return G


def test_normalized_adjacency_is_symmetric_renormalized():
    # Â = D̃^{-1/2}(A+I)D̃^{-1/2}. For a single edge a-b: A+I = [[1,1],[1,1]], d=2,
    # so Â = 0.5 * ones(2,2).
    G = nx.Graph(); G.add_edge("a", "b", weight=1.0)
    nodes, Ahat = normalized_adjacency(G)
    assert np.allclose(Ahat, 0.5 * np.ones((2, 2)))
    # symmetric on a bigger graph
    _, Ah = normalized_adjacency(_bridged_triangles())
    assert np.allclose(Ah, Ah.T)


def test_sgc_propagate_smooths_over_neighborhoods():
    # a within-community node's propagated feature should move toward its neighbors'
    G = _bridged_triangles()
    nodes, Ahat = normalized_adjacency(G)
    X = np.eye(len(nodes))
    H = sgc_propagate(Ahat, X, k=2)
    i = {n: k for k, n in enumerate(nodes)}
    # after 2 hops, a and c (same triangle) are more similar than a and z (far)
    d_near = np.linalg.norm(H[i["a"]] - H[i["c"]])
    d_far = np.linalg.norm(H[i["a"]] - H[i["z"]])
    assert d_far > d_near


def test_sgc_recovers_planted_communities_semi_supervised():
    G = _bridged_triangles()
    nodes, Ahat = normalized_adjacency(G)
    X = np.eye(len(nodes))                       # one-hot structural features
    comm = {"a": 0, "b": 0, "c": 0, "x": 1, "y": 1, "z": 1}
    y = np.array([comm[n] for n in nodes])
    labeled = [nodes.index("a"), nodes.index("x")]   # one seed per community
    W = train_sgc(Ahat, X, y, labeled, k=2, n_classes=2, epochs=400, lr=0.5, seed=0)
    pred = sgc_predict(Ahat, X, W, k=2)
    got = {nodes[j]: int(pred[j]) for j in range(len(nodes))}
    assert got["a"] == got["b"] == got["c"]      # community 1 recovered
    assert got["x"] == got["y"] == got["z"]      # community 2 recovered
    assert got["a"] != got["x"]


def test_sgc_gpu_matches_cpu_predictions():
    pytest.importorskip("cupy")     # skips off-box; runs on a CuPy/GPU box
    G = _bridged_triangles()
    nodes, Ahat = normalized_adjacency(G)
    X = np.eye(len(nodes))
    comm = {"a": 0, "b": 0, "c": 0, "x": 1, "y": 1, "z": 1}
    y = np.array([comm[n] for n in nodes])
    labeled = [nodes.index("a"), nodes.index("x")]
    kw = dict(k=2, n_classes=2, epochs=400, lr=0.5, seed=0)
    Wc = train_sgc(Ahat, X, y, labeled, backend="cpu", **kw)
    Wg = train_sgc(Ahat, X, y, labeled, backend="gpu", **kw)
    pc = sgc_predict(Ahat, X, Wc, k=2, backend="cpu")
    pg = sgc_predict(Ahat, X, Wg, k=2, backend="gpu")
    assert np.array_equal(pc, pg)   # cpu and gpu SGC agree on the labels
