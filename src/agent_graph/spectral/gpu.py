"""Optional GPU (CuPy) backend for the Laplacian eigendecomposition.

Lazy-imported and gated behind the ``[gpu]`` extra so CPU-only installs / CI stay
GPU-free. Every function returns **host numpy** (via ``cupy.asnumpy``) so callers
stay device-agnostic and the return contract is identical to the scipy path.

Why CuPy and not cuGraph for the solve: cuGraph builds graphs on-device and runs
spectral *clustering*, but does not expose the raw eigenpairs the spectral layer
returns — so the eigen-solve itself is CuPy dense ``eigh`` (exact parity with the
CPU dense path). A sparse CuPy ``eigsh`` for very large graphs is a follow-up; the
dense path is correct and GPU-accelerated for the graph sizes here.
"""
from __future__ import annotations

import numpy as np


def gpu_available() -> bool:
    """True iff CuPy is importable (the GPU box). Never raises."""
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    """'auto' -> 'gpu' if CuPy is importable, else 'cpu'. 'cpu'/'gpu' pass through."""
    if backend == "auto":
        return "gpu" if gpu_available() else "cpu"
    return backend


def louvain_gpu(G, resolution: float = 1.0) -> dict:
    """cuGraph Louvain community detection -> ``dict[node -> community_id]``, the
    same contract as the CPU path. NOTE: this is a *different algorithm instance*
    than networkx Louvain (both are stochastic / implementation-specific), so it
    recovers the same community STRUCTURE on well-separated graphs, not identical
    labels. Lazy-imports cuGraph/cuDF; validated on-box (the parity test skips
    off-box). API is cuGraph-version-sensitive (partition column / resolution)."""
    import cudf
    import cugraph

    nodes = list(G.nodes())
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(nodes)}     # each node its own community
    idx = {n: i for i, n in enumerate(nodes)}
    src = [idx[u] for u, v in G.edges()]
    dst = [idx[v] for u, v in G.edges()]
    wts = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]
    df = cudf.DataFrame({"src": src, "dst": dst, "weight": wts})
    Gc = cugraph.Graph()                                # undirected
    Gc.from_cudf_edgelist(df, source="src", destination="dst",
                          edge_attr="weight", renumber=True)
    parts, _modularity = cugraph.louvain(Gc, resolution=resolution)
    pdf = parts.to_pandas()
    vertex_to_part = dict(zip(pdf["vertex"], pdf["partition"]))
    return {nodes[v]: int(vertex_to_part[v]) for v in range(len(nodes))}


def small_eigs_gpu(L, k: int) -> np.ndarray:
    """Smallest ``k+1`` Laplacian eigenvalues (ascending), as host numpy — CuPy
    dense ``eigvalsh``. Mirrors the CPU dense branch of ``_small_eigs``."""
    import cupy as cp
    w = cp.linalg.eigvalsh(cp.asarray(L.toarray()))
    return cp.asnumpy(cp.sort(w)[: k + 1])


def spectral_embedding_gpu(L, k: int) -> np.ndarray:
    """Laplacian-eigenmap coordinates (host numpy): drop the trivial constant
    eigenvector, take the next ``k`` — CuPy dense ``eigh``. Mirrors the CPU dense
    branch of ``spectral_embedding``."""
    import cupy as cp
    w, V = cp.linalg.eigh(cp.asarray(L.toarray()))
    order = cp.argsort(w)
    coords = V[:, order][:, 1 : k + 1]
    return cp.asnumpy(coords)
