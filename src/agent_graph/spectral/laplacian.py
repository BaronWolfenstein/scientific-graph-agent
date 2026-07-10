"""Combinatorial graph Laplacian L = D - A and its low eigenvectors.

CPU via scipy.sparse.linalg.eigsh (default). An optional GPU backend (CuPy dense
eigh, `spectral.gpu`, gated behind the `[gpu]` extra) is selected with
`backend="gpu"` (or `"auto"` → GPU if CuPy is importable, else CPU); it returns
host numpy so the return contract is identical. Existing callers are unaffected
(default `backend="cpu"` reproduces the prior behavior byte-for-byte).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def combinatorial_laplacian(G: nx.Graph) -> tuple[list[str], sp.csr_matrix]:
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(d) - A
    return nodes, L.tocsr()


from .gpu import resolve_backend as _resolve_backend  # noqa: E402


def _small_eigs(L: sp.csr_matrix, k: int, backend: str = "cpu") -> np.ndarray:
    if backend == "gpu":
        from .gpu import small_eigs_gpu
        return small_eigs_gpu(L, k)
    n = L.shape[0]
    k_eff = min(k, n - 1)
    # shift-invert near 0 for the smallest eigenvalues; dense fallback for tiny graphs
    if n <= 12:
        w = np.linalg.eigvalsh(L.toarray())
        return np.sort(w)[: k + 1]
    vals = spla.eigsh(L, k=k_eff + 1, sigma=0, which="LM", return_eigenvectors=False)
    return np.sort(vals)


def spectral_gap(G: nx.Graph, k: int = 8, backend: str = "cpu") -> list[float]:
    _, L = combinatorial_laplacian(G)
    return [float(v) for v in _small_eigs(L, k, _resolve_backend(backend))]


def spectral_embedding(G: nx.Graph, k: int = 8,
                       backend: str = "cpu") -> tuple[list[str], np.ndarray]:
    nodes, L = combinatorial_laplacian(G)
    if _resolve_backend(backend) == "gpu":
        from .gpu import spectral_embedding_gpu
        return nodes, spectral_embedding_gpu(L, k)
    n = L.shape[0]
    k_eff = min(k + 1, n)
    if n <= 12:
        w, V = np.linalg.eigh(L.toarray())
    else:
        w, V = spla.eigsh(L, k=k_eff, sigma=0, which="LM")
    order = np.argsort(w)
    # drop the trivial constant eigenvector (eigenvalue 0); take next k
    coords = V[:, order][:, 1 : k + 1]
    return nodes, np.asarray(coords)
