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


def sgc_propagate(Ahat: np.ndarray, X: np.ndarray, k: int = 2) -> np.ndarray:
    """k-hop spectral feature propagation ``Â^k X`` (the SGC feature map)."""
    H = np.asarray(X, dtype=float)
    for _ in range(k):
        H = Ahat @ H
    return H


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def train_sgc(Ahat, X, y, labeled_idx, *, k: int = 2, n_classes=None,
              epochs: int = 300, lr: float = 0.5, l2: float = 1e-3,
              seed: int = 0) -> np.ndarray:
    """Train the SGC softmax head on the ``labeled_idx`` nodes only (semi-supervised
    node classification). Features are propagated once (``Â^k X``); the classifier
    is L2-regularized softmax regression trained by analytic gradient descent.
    Returns weights ``W`` of shape ``(feat_dim, n_classes)``."""
    H = sgc_propagate(Ahat, X, k)                   # (n, d)
    y = np.asarray(y)
    labeled_idx = np.asarray(labeled_idx)
    if n_classes is None:
        n_classes = int(y[labeled_idx].max()) + 1
    d = H.shape[1]
    rng = np.random.default_rng(seed)
    W = 0.01 * rng.standard_normal((d, n_classes))
    Hl = H[labeled_idx]                             # (m, d)
    Yl = np.zeros((len(labeled_idx), n_classes))
    Yl[np.arange(len(labeled_idx)), y[labeled_idx]] = 1.0
    for _ in range(epochs):
        P = _softmax(Hl @ W)                        # (m, C)
        grad = Hl.T @ (P - Yl) / len(labeled_idx) + l2 * W
        W -= lr * grad
    return W


def sgc_predict(Ahat, X, W, k: int = 2) -> np.ndarray:
    """Predicted class per node: ``argmax softmax(Â^k X · W)``."""
    H = sgc_propagate(Ahat, X, k)
    return _softmax(H @ W).argmax(axis=1)
