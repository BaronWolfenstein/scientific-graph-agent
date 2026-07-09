"""Optimal transport over the spectral graph layer (additive, CPU-only).

Two complementary distances, no new dependency (numpy + scipy + the existing
Laplacian):

- ``sinkhorn_divergence`` — debiased entropic OT between two point clouds in a
  *semantic* feature space ("match in semantic space"; a tractable Wasserstein
  surrogate for distribution matching / shift).
- ``gromov_wasserstein`` + ``heat_kernel_cost`` / ``graph_gw_distance`` — compare
  two graphs by *structure alone* (no shared node space), bridging the Laplacian
  diffusion geometry to Gromov-Wasserstein.

All solvers are log-domain / entropic and run on the small KG subgraphs the
spectral layer targets.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import logsumexp

from .laplacian import combinatorial_laplacian


def _sqdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean cost matrix (n, m)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)


def _sinkhorn_plan(M: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float,
                   max_iter: int, tol: float) -> np.ndarray:
    """Entropic-OT coupling for cost ``M`` and marginals ``a``, ``b`` (log-domain).

    Potentials f, g satisfy the balanced fixed point; the returned plan has row
    marginal ``a`` and column marginal ``b`` at convergence.
    """
    loga = np.log(a)
    logb = np.log(b)
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    for _ in range(max_iter):
        f_new = -eps * logsumexp((g[None, :] - M) / eps + logb[None, :], axis=1)
        g = -eps * logsumexp((f_new[:, None] - M) / eps + loga[:, None], axis=0)
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    logP = (f[:, None] + g[None, :] - M) / eps + loga[:, None] + logb[None, :]
    return np.exp(logP)


def _sinkhorn_cost(M, a, b, eps, max_iter, tol) -> float:
    P = _sinkhorn_plan(M, a, b, eps, max_iter, tol)
    return float(np.sum(P * M))


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def sinkhorn_divergence(X: np.ndarray, Y: np.ndarray, *, eps: float = 0.1,
                        max_iter: int = 500, tol: float = 1e-9) -> float:
    """Debiased Sinkhorn divergence between point clouds ``X`` (n, d), ``Y`` (m, d).

    ``S = OT_eps(X,Y) - ½ OT_eps(X,X) - ½ OT_eps(Y,Y)`` (squared-Euclidean cost,
    uniform marginals). Debiasing gives ``S(X,X) = 0`` and ``S >= 0``; a tractable
    Wasserstein surrogate for matching two distributions in a semantic space.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    a = _uniform(len(X))
    b = _uniform(len(Y))
    oxy = _sinkhorn_cost(_sqdist(X, Y), a, b, eps, max_iter, tol)
    oxx = _sinkhorn_cost(_sqdist(X, X), a, a, eps, max_iter, tol)
    oyy = _sinkhorn_cost(_sqdist(Y, Y), b, b, eps, max_iter, tol)
    return oxy - 0.5 * oxx - 0.5 * oyy


def heat_kernel_cost(G, *, t: float = 1.0, k: int = 8):
    """Structural cost matrix from the Laplacian diffusion geometry.

    Returns ``(nodes, C)`` where ``C[x, y] = sum_i e^{-2 t lam_i} (phi_i(x) -
    phi_i(y))^2`` is the squared diffusion distance at time ``t`` over the ``k``
    smallest Laplacian eigenpairs (the modes with the largest diffusion weight).
    This is the intra-graph cost GW consumes — the spectrum→OT bridge. The
    trivial constant eigenvector contributes zero and is harmless to include.
    """
    nodes, L = combinatorial_laplacian(G)
    n = L.shape[0]
    k_eff = min(k, n)
    if n <= 12:
        w, V = np.linalg.eigh(L.toarray())          # ascending
        w, V = w[:k_eff], V[:, :k_eff]
    else:
        w, V = spla.eigsh(L, k=k_eff, sigma=0, which="LM")
        order = np.argsort(w)
        w, V = w[order], V[:, order]
    psi = V * np.exp(-t * w)[None, :]               # weighted diffusion map
    C = ((psi[:, None, :] - psi[None, :, :]) ** 2).sum(axis=2)
    C = 0.5 * (C + C.T)                             # kill fp asymmetry
    np.fill_diagonal(C, 0.0)
    return nodes, C


def gromov_wasserstein(C1: np.ndarray, C2: np.ndarray, p=None, q=None, *,
                       eps: float = 0.05, max_iter: int = 1000, tol: float = 1e-9,
                       sink_iter: int = 200, sink_tol: float = 1e-9):
    """Entropic Gromov-Wasserstein between intra-graph cost matrices ``C1`` (n, n)
    and ``C2`` (m, m) (Peyré et al. 2016, square loss).

    Compares two graphs by *structure alone* — no shared node space. Returns
    ``(T, gw_cost)``: the soft coupling (marginals ``p``, ``q``) and the GW
    energy ``E(T) = <constC - 2 C1 T C2, T>``. Solved by block-coordinate descent
    with a log-domain Sinkhorn projection each step.
    """
    C1 = np.asarray(C1, float)
    C2 = np.asarray(C2, float)
    n, m = C1.shape[0], C2.shape[0]
    p = _uniform(n) if p is None else np.asarray(p, float)
    q = _uniform(m) if q is None else np.asarray(q, float)
    # constC_ij = sum_k C1_ik^2 p_k + sum_l C2_jl^2 q_l  (the marginal-fixed part)
    constC = ((C1 ** 2) @ p)[:, None] + ((C2 ** 2) @ q)[None, :]
    T = np.outer(p, q)
    for _ in range(max_iter):
        tens = constC - 2.0 * (C1 @ T @ C2)         # loss tensor L (x) T
        T_new = _sinkhorn_plan(tens, p, q, eps, sink_iter, sink_tol)
        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break
        T = T_new
    tens = constC - 2.0 * (C1 @ T @ C2)
    gw = float(np.sum(tens * T))
    return T, gw


def graph_gw_distance(G1, G2, *, t: float = 0.5, k: int = 8, eps: float = 0.05,
                      max_iter: int = 1000, tol: float = 1e-9) -> float:
    """Debiased Gromov-Wasserstein distance between two graphs, by structure alone.

    Builds each graph's heat-kernel (diffusion) cost, then returns the debiased
    GW divergence ``GW(C1,C2) - ½ GW(C1,C1) - ½ GW(C2,C2)``. Debiasing cancels the
    entropic-regularization bias so the distance is ~0 for isometric (relabeled)
    graphs and positive for structurally distinct ones. The graph-facing
    "structural intelligence" metric.
    """
    _, C1 = heat_kernel_cost(G1, t=t, k=k)
    _, C2 = heat_kernel_cost(G2, t=t, k=k)
    kw = dict(eps=eps, max_iter=max_iter, tol=tol)
    _, g12 = gromov_wasserstein(C1, C2, **kw)
    _, g11 = gromov_wasserstein(C1, C1, **kw)
    _, g22 = gromov_wasserstein(C2, C2, **kw)
    return g12 - 0.5 * g11 - 0.5 * g22
