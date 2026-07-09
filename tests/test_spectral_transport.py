"""Optimal-transport layer over the spectral graph representation.

Sinkhorn divergence (semantic/embedding space) + Gromov-Wasserstein (pure
structure) + a heat-kernel structural cost that bridges the Laplacian spectrum
to GW. CPU numpy/scipy, no new dependency.
"""
import numpy as np
import networkx as nx

from agent_graph.spectral.transport import (
    sinkhorn_divergence,
    heat_kernel_cost,
    gromov_wasserstein,
    graph_gw_distance,
)


def _bridged_triangles():
    """Two triangles joined by a single bridge edge (b-x)."""
    G = nx.Graph()
    for a, b in [("a", "b"), ("b", "c"), ("a", "c"),
                 ("x", "y"), ("y", "z"), ("x", "z"),
                 ("b", "x")]:
        G.add_edge(a, b, weight=1.0)
    return G


# ---- Sinkhorn divergence (point clouds in a semantic space) ----

def test_sinkhorn_divergence_of_a_cloud_with_itself_is_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    s = sinkhorn_divergence(X, X, eps=0.1)
    assert abs(s) < 1e-6                    # debiased: S(X,X) = 0 exactly


def test_sinkhorn_divergence_is_symmetric():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((25, 3))
    Y = rng.standard_normal((20, 3)) + 1.0
    assert np.isclose(sinkhorn_divergence(X, Y, eps=0.1),
                      sinkhorn_divergence(Y, X, eps=0.1), atol=1e-9)


def test_sinkhorn_divergence_grows_as_clouds_separate():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 3))
    near = sinkhorn_divergence(X, X + 0.5, eps=0.1)
    far = sinkhorn_divergence(X, X + 4.0, eps=0.1)
    assert far > near > 0                   # monotone in separation, non-negative


# ---- heat-kernel structural cost (Laplacian diffusion geometry) ----

def test_heat_kernel_cost_is_symmetric_zero_diagonal_nonneg():
    G = _bridged_triangles()
    nodes, C = heat_kernel_cost(G, t=0.5)
    assert C.shape == (len(nodes), len(nodes))
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 0.0)
    assert (C >= -1e-9).all()


def test_heat_kernel_cost_matches_diffusion_distance_formula():
    import networkx as nx
    G = _bridged_triangles()
    nodes, C = heat_kernel_cost(G, t=0.7, k=6)
    # direct diffusion distance d_t(x,y)^2 = sum_i e^{-2 t lam_i} (phi_i(x)-phi_i(y))^2
    L = nx.laplacian_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)
    w, V = np.linalg.eigh(L)
    psi = V * np.exp(-0.7 * w)[None, :]
    D = ((psi[:, None, :] - psi[None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(C, D, atol=1e-6)


def test_heat_kernel_within_cluster_closer_than_across_bridge():
    G = _bridged_triangles()
    nodes, C = heat_kernel_cost(G, t=0.5)
    i = {n: k for k, n in enumerate(nodes)}
    within = C[i["a"], i["c"]]          # same triangle
    across = C[i["a"], i["z"]]          # opposite triangle, over the bridge
    assert across > within


# ---- Gromov-Wasserstein (compare graphs by structure alone) ----

def test_gw_coupling_respects_marginals():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5)); C1 = np.abs(A + A.T)
    B = rng.standard_normal((4, 4)); C2 = np.abs(B + B.T)
    np.fill_diagonal(C1, 0.0); np.fill_diagonal(C2, 0.0)
    T, _ = gromov_wasserstein(C1, C2, eps=0.05)
    assert np.allclose(T.sum(axis=1), np.full(5, 1 / 5), atol=1e-3)
    assert np.allclose(T.sum(axis=0), np.full(4, 1 / 4), atol=1e-3)


def test_gw_energy_is_invariant_to_relabeling():
    # raw entropic GW carries a regularization bias (not 0 for isometric graphs),
    # but its energy must be invariant to permuting either graph's node labels.
    G = _bridged_triangles()
    _, C = heat_kernel_cost(G, t=0.5)
    perm = np.array([3, 0, 5, 1, 4, 2])
    C_perm = C[np.ix_(perm, perm)]
    _, gw_self = gromov_wasserstein(C, C, eps=0.02)
    _, gw_perm = gromov_wasserstein(C, C_perm, eps=0.02)
    assert np.isclose(gw_self, gw_perm, atol=1e-3)


def test_gw_larger_for_structurally_different_graphs():
    import networkx as nx
    path = nx.path_graph(5)
    star = nx.star_graph(4)               # 1 center + 4 leaves = 5 nodes
    nx.set_edge_attributes(path, 1.0, "weight")
    nx.set_edge_attributes(star, 1.0, "weight")
    _, Cp = heat_kernel_cost(path, t=0.5)
    _, Cs = heat_kernel_cost(star, t=0.5)
    _, gw_same = gromov_wasserstein(Cp, Cp, eps=0.02)
    _, gw_diff = gromov_wasserstein(Cp, Cs, eps=0.02)
    assert gw_diff > gw_same


# ---- graph_gw_distance: debiased, graph-facing structural distance ----

def test_graph_gw_distance_of_relabeled_graph_is_near_zero():
    import networkx as nx
    G = _bridged_triangles()
    mapping = {"a": "n1", "b": "n2", "c": "n3", "x": "n4", "y": "n5", "z": "n6"}
    H = nx.relabel_nodes(G, mapping)
    d = graph_gw_distance(G, H, t=0.5, eps=0.02)
    assert abs(d) < 1e-3                   # debiasing cancels the entropic bias


def test_graph_gw_distance_is_positive_for_different_structure():
    import networkx as nx
    path = nx.path_graph(5); nx.set_edge_attributes(path, 1.0, "weight")
    star = nx.star_graph(4); nx.set_edge_attributes(star, 1.0, "weight")
    d = graph_gw_distance(path, star, t=0.5, eps=0.02)
    assert d > 1e-2                        # path and star are structurally distinct


def test_graph_gw_distance_is_symmetric():
    import networkx as nx
    path = nx.path_graph(5); nx.set_edge_attributes(path, 1.0, "weight")
    star = nx.star_graph(4); nx.set_edge_attributes(star, 1.0, "weight")
    assert np.isclose(graph_gw_distance(path, star, t=0.5, eps=0.02),
                      graph_gw_distance(star, path, t=0.5, eps=0.02), atol=1e-6)
