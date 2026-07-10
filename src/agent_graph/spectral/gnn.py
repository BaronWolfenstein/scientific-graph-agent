"""Spectral graph neural network over the entity graph — Simplified Graph
Convolution (SGC, Wu et al. 2019, "Simplifying Graph Convolutional Networks").

SGC is the linearized/spectral core of a GCN (Kipf & Welling 2017): propagate the
node features with the k-hop **renormalized adjacency** ``Â^k`` (``Â =
D̃^{-1/2}(A+I)D̃^{-1/2}``, a first-order spectral graph filter), then a plain
softmax classifier. Dropping the intermediate nonlinearities makes it a cheap,
convex, strong node-classification baseline that reuses the same graph-Laplacian
machinery as the rest of the spectral layer. numpy/CPU; the softmax head trains by
analytic-gradient descent. A torch / cuGraph-GNN backend is a follow-up.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def normalized_adjacency(G: nx.Graph, nodes=None) -> tuple[list, np.ndarray]:
    """Symmetric renormalized adjacency ``Â = D̃^{-1/2}(A+I)D̃^{-1/2}`` (the
    Kipf-Welling propagation operator; self-loops added). Returns ``(nodes, Â)``."""
    nodes = list(G.nodes()) if nodes is None else list(nodes)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight",
                                 format="csr").toarray().astype(float)
    A = A + np.eye(len(nodes))                      # add self-loops -> Ã
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    return nodes, d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def _xp_for(backend: str):
    """Array namespace for the SGC matmuls: cupy on the GPU box (``backend='gpu'``
    or ``'auto'`` when CuPy is present), else numpy. Lazy CuPy import."""
    from .gpu import resolve_backend
    if resolve_backend(backend) == "gpu":
        import cupy as cp
        return cp
    return np


def _to_host(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)   # cupy -> numpy


def _softmax(Z, xp):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = xp.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def sgc_propagate(Ahat, X, k: int = 2, backend: str = "cpu") -> np.ndarray:
    """k-hop spectral feature propagation ``Â^k X`` (the SGC feature map). Runs the
    matmuls on GPU when ``backend='gpu'``; always returns host numpy."""
    xp = _xp_for(backend)
    H = xp.asarray(X, dtype=float)
    Ah = xp.asarray(Ahat, dtype=float)
    for _ in range(k):
        H = Ah @ H
    return _to_host(H)


def train_sgc(Ahat, X, y, labeled_idx, *, k: int = 2, n_classes=None,
              epochs: int = 300, lr: float = 0.5, l2: float = 1e-3,
              seed: int = 0, backend: str = "cpu") -> np.ndarray:
    """Train the SGC softmax head on the ``labeled_idx`` nodes only (semi-supervised
    node classification). Features are propagated once (``Â^k X``); the classifier
    is L2-regularized softmax regression trained by analytic gradient descent. The
    propagation and training matmuls run on GPU when ``backend='gpu'``; returns
    host-numpy weights ``W`` of shape ``(feat_dim, n_classes)``."""
    xp = _xp_for(backend)
    y = np.asarray(y)
    labeled_idx = np.asarray(labeled_idx)
    if n_classes is None:
        n_classes = int(y[labeled_idx].max()) + 1
    Ah = xp.asarray(Ahat, dtype=float)
    H = xp.asarray(X, dtype=float)
    for _ in range(k):
        H = Ah @ H                                  # propagate Â^k X on-device
    Hl = H[xp.asarray(labeled_idx)]                 # (m, d)
    d = Hl.shape[1]
    W = xp.asarray(0.01 * np.random.default_rng(seed).standard_normal((d, n_classes)))
    Yl = xp.zeros((len(labeled_idx), n_classes))
    Yl[xp.arange(len(labeled_idx)), xp.asarray(y[labeled_idx])] = 1.0
    for _ in range(epochs):
        P = _softmax(Hl @ W, xp)                    # (m, C)
        grad = Hl.T @ (P - Yl) / len(labeled_idx) + l2 * W
        W = W - lr * grad
    return _to_host(W)


def sgc_predict(Ahat, X, W, k: int = 2, backend: str = "cpu") -> np.ndarray:
    """Predicted class per node: ``argmax softmax(Â^k X · W)`` (host numpy out)."""
    xp = _xp_for(backend)
    H = xp.asarray(X, dtype=float)
    Ah = xp.asarray(Ahat, dtype=float)
    for _ in range(k):
        H = Ah @ H
    pred = _softmax(H @ xp.asarray(W, dtype=float), xp).argmax(axis=1)
    return _to_host(pred)
