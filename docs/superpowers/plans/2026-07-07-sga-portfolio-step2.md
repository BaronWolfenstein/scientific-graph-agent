# SGA Portfolio Step 2 — Program Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `scientific-graph-agent` into a portfolio centrepiece that demonstrates the skills the Mount Sinai (LLM Engineer) and Scier (Structural Intelligence) roles require but that no current work covers — spectral graph computation, QLoRA fine-tuning, RAG/eval at MVP, and causal edge-validation — layered additively on the existing LangGraph agent + (planned) RDF-star knowledge graph.

**Architecture:** Additive packages under `src/agent_graph/`, each behind a small Protocol, each independently testable. The spectral layer consumes the KG's `pyoxigraph` Store directly (stable `quads_for_pattern` API) and writes versioned spectral snapshots back as first-class KG nodes. Nothing rewrites the flat-paper pipeline.

**Tech Stack:** Python 3.11, LangGraph, `pyoxigraph` (RDF-star Store), `networkx` + `scipy` (spectral, CPU), `python-igraph`/`leidenalg` (Leiden, optional), `cugraph` (GPU, deferred), `peft`/`transformers`/`bitsandbytes` (QLoRA, deferred), `gradio` (MVP), `deepeval` (eval, existing plans), pytest.

## Global Constraints

- Python `>=3.11` (matches existing `pyproject.toml`).
- **Additive only** — do not modify the flat-paper pipeline (`nodes.py`, `state.py`, `graph.py`) except to register new optional nodes. Same rule the KG and assurance plans follow.
- All new packages sit under `src/agent_graph/<pkg>/` behind a Protocol; wrap third-party libs (never import `cugraph`/`peft` at module top level — lazy-import inside functions so CPU-only test runs don't need GPU deps).
- Every new dependency added to `pyproject.toml` `dependencies` with an explicit lower bound.
- TDD: failing test → minimal impl → green → commit. Frequent commits.
- Fork remote: push to `origin` = `BaronWolfenstein/scientific-graph-agent`; `gh` defaults to `upstream`, so always pass `-R BaronWolfenstein/scientific-graph-agent`.

---

## Program overview — what exists, what's net-new

**Substrate already planned (do NOT re-plan here; these are prerequisites):**
- `docs/superpowers/plans/2026-06-27-domain-knowledge-graph.md` — the RDF-star claim graph in `src/agent_graph/kg/` (pyoxigraph Store, reifier model with confidence/evidence/provenance). **Prerequisite for the spectral layer and zero-flow CI.** Currently plan-only (package not built).
- `docs/superpowers/plans/2026-07-03-assurance-harness-phase1-core.md` + `2026-07-04-assurance-harness-phase2-deepeval.md` + security phase — the eval harness (`src/agent_graph/assurance/`, DeepEval, RAG triad, clinical metrics). **This is the Mount Sinai eval-pipeline subsystem, already specced.** Currently plan-only.
- Existing built code: LangGraph agent (`agent_graph/`), ArXiv+PubMed retrieval (`tools.py`), reranker, GEPA prompt-optimisation (`optimize/`), faithfulness/consistency eval (`eval/`).

**Net-new subsystems this program adds** (each becomes its own plan — see Scope Check), mapped to the role each serves:

| # | Subsystem | Serves | Depends on | GPU |
|---|---|---|---|---|
| **A** | **Spectral graph layer** (`spectral/`) — Laplacian `L=D−A`, spectral embedding, Leiden/Louvain communities, centrality, snapshots-as-KG-nodes | **Scier (core)** | KG plan | no (CPU; cuGraph later) |
| B | Zero-flow CI edge validation (`kg/verify/`) — attach data-driven confidence reifiers to causal edges | Scier + Alembic (hedge) | KG plan, data | no |
| C | QLoRA fine-tune (`finetune/`) — graph-aware SFT of a small base model on KG-structured claims | **Mount Sinai + Scier** | claims corpus | **yes** |
| D | Gradio MVP (`app/`) — surface retrieval + communities + spectral snapshots to a user | Mount Sinai + Scier | A (+ existing retrieval) | no |

## Scope Check

This program spans four independent subsystems. **Each MUST be its own plan** (own spec → own `docs/superpowers/plans/` file), because each produces working, testable software on its own and a reviewer could accept one while rejecting another. This document fully details **Subsystem A (Spectral graph layer)** — the foundational net-new piece: it's Scier's core skill gap, it's CPU-only (buildable now), it operates on the existing KG, and Subsystem D (MVP) consumes it. B/C/D are outlined at the end and get their own plans when reached.

**Recommended build order:** execute the KG plan first (prerequisite) → **A (spectral)** → D (MVP, so there's something demonstrable) → C (QLoRA, when GPU is available) → B (zero-flow CI, when a suitable dataset is in hand).

---

# Subsystem A — Spectral Graph Layer (fully detailed)

**Prerequisite:** the domain-KG plan is executed, so `src/agent_graph/kg/` exists and a populated `pyoxigraph.Store` is available. The spectral layer depends only on pyoxigraph's stable `Store.quads_for_pattern(subject, predicate, object, graph_name)` API, not on the KG wrapper's internal names — so it is robust to KG-package naming.

**Files:**
- Create: `src/agent_graph/spectral/__init__.py`
- Create: `src/agent_graph/spectral/graph_adapter.py` — Store → `networkx.Graph`
- Create: `src/agent_graph/spectral/laplacian.py` — Laplacian + spectral embedding + gap
- Create: `src/agent_graph/spectral/communities.py` — Louvain (default) / Leiden (optional)
- Create: `src/agent_graph/spectral/centrality.py` — betweenness + bridging centrality
- Create: `src/agent_graph/spectral/snapshots.py` — versioned spectral snapshot → KG nodes
- Create: `tests/test_spectral_adapter.py`, `tests/test_spectral_laplacian.py`, `tests/test_spectral_communities.py`, `tests/test_spectral_centrality.py`, `tests/test_spectral_snapshots.py`
- Modify: `pyproject.toml` (add `networkx>=3.2`, `scipy>=1.11`; `python-igraph>=0.11` + `leidenalg>=0.10` as optional extras)

**Interfaces:**
- Consumes: a `pyoxigraph.Store` populated by the KG package (RDF-star claims; entities are triple subjects/objects, claim predicates are the edges).
- Produces:
  - `build_entity_graph(store, *, predicates=None, weight_by_confidence=True) -> networkx.Graph`
  - `combinatorial_laplacian(G) -> tuple[list[str], scipy.sparse.csr_matrix]`  (node order, L)
  - `spectral_embedding(G, k=8) -> tuple[list[str], numpy.ndarray]`  (node order, (n,k) Fiedler coords)
  - `spectral_gap(G, k=8) -> list[float]`  (first k+1 eigenvalues)
  - `detect_communities(G, method="louvain", resolution=1.0, seed=0) -> dict[str, int]`
  - `bridging_centrality(G) -> dict[str, float]`
  - `SpectralSnapshot` dataclass + `write_spectral_snapshot(store, snapshot, *, graph_name=...) -> str` (snapshot IRI)

---

### Task A1: dependencies + package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/agent_graph/spectral/__init__.py`

- [ ] **Step 1: Add deps.** In `pyproject.toml` `dependencies`, add `"networkx>=3.2"` and `"scipy>=1.11"`. Add an optional-extras table if not present:

```toml
[project.optional-dependencies]
leiden = ["python-igraph>=0.11", "leidenalg>=0.10"]
gpu = ["cugraph-cu12>=24.06"]
```

- [ ] **Step 2: Create the package `__init__`** exporting the public names:

```python
"""Spectral graph layer over the RDF-star knowledge graph (additive)."""
from .graph_adapter import build_entity_graph
from .laplacian import combinatorial_laplacian, spectral_embedding, spectral_gap
from .communities import detect_communities
from .centrality import bridging_centrality
from .snapshots import SpectralSnapshot, write_spectral_snapshot

__all__ = [
    "build_entity_graph", "combinatorial_laplacian", "spectral_embedding",
    "spectral_gap", "detect_communities", "bridging_centrality",
    "SpectralSnapshot", "write_spectral_snapshot",
]
```

- [ ] **Step 3: Install + import-check.** Run: `pip install -e . && python -c "import networkx, scipy; print('ok')"` — Expected: `ok`. (The package `__init__` will fail to import until later tasks create the modules; that is expected now.)
- [ ] **Step 4: Commit.**

```bash
git add pyproject.toml src/agent_graph/spectral/__init__.py
git commit -m "feat(spectral): package skeleton + networkx/scipy deps"
```

---

### Task A2: Store → NetworkX adapter

**Files:**
- Create: `src/agent_graph/spectral/graph_adapter.py`
- Test: `tests/test_spectral_adapter.py`

**Interfaces:**
- Produces: `build_entity_graph(store, *, predicates=None, weight_by_confidence=True) -> networkx.Graph`

- [ ] **Step 1: Write the failing test.** Build a tiny in-memory pyoxigraph Store with three entity→entity claim triples and assert the graph shape.

```python
import networkx as nx
import pyoxigraph as ox
from agent_graph.spectral.graph_adapter import build_entity_graph

def _triple(s, p, o):
    return ox.Quad(ox.NamedNode(s), ox.NamedNode(p), ox.NamedNode(o))

def test_build_entity_graph_nodes_and_edges():
    store = ox.Store()
    E = "http://ex/entity/"; P = "http://ex/claim/affects"
    store.add(_triple(E + "A", P, E + "B"))
    store.add(_triple(E + "B", P, E + "C"))
    store.add(_triple(E + "A", P, E + "C"))
    G = build_entity_graph(store)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3
    assert G.has_edge(E + "A", E + "B")
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_spectral_adapter.py -q` — Expected: FAIL (`ModuleNotFoundError` / `build_entity_graph` undefined).
- [ ] **Step 3: Implement the adapter.**

```python
"""Build an undirected NetworkX graph of entities from the KG Store.

Entities are triple subjects/objects; claim predicates are edges. Edge weight
is the accumulated Beta-Bernoulli confidence on the claim's reifier when
available (the KG plan stores confidence as a reifier annotation), else 1.0.
Undirected + combinatorial because the Laplacian spectrum is defined on the
symmetric adjacency; direction is preserved in the KG itself, not here.
"""
from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx
import pyoxigraph as ox

# Predicates that are structural (rdf:type, reifies, provenance) and must NOT
# become graph edges. Extend as the KG ontology grows.
_NON_EDGE_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies",
}


def build_entity_graph(
    store: ox.Store,
    *,
    predicates: Optional[Iterable[str]] = None,
    weight_by_confidence: bool = True,
) -> nx.Graph:
    allow = set(predicates) if predicates is not None else None
    G = nx.Graph()
    for quad in store.quads_for_pattern(None, None, None, None):
        p = quad.predicate.value
        if p in _NON_EDGE_PREDICATES:
            continue
        if allow is not None and p not in allow:
            continue
        s, o = quad.subject, quad.object
        # entity→entity edges only (both endpoints are NamedNodes)
        if not (isinstance(s, ox.NamedNode) and isinstance(o, ox.NamedNode)):
            continue
        w = 1.0
        if G.has_edge(s.value, o.value):
            w += G[s.value][o.value].get("weight", 1.0)
        G.add_edge(s.value, o.value, weight=w, predicate=p)
    return G
```

*(Note: `weight_by_confidence` is wired as a parameter now; reifier-confidence lookup is added in Task B's plan, where the reifier read-path is defined. Until then weight = multiplicity, which is a valid structural weight.)*

- [ ] **Step 4: Run to verify it passes.** Run: `pytest tests/test_spectral_adapter.py -q` — Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/agent_graph/spectral/graph_adapter.py tests/test_spectral_adapter.py
git commit -m "feat(spectral): pyoxigraph Store -> networkx entity graph"
```

---

### Task A3: Laplacian, spectral embedding, spectral gap

**Files:**
- Create: `src/agent_graph/spectral/laplacian.py`
- Test: `tests/test_spectral_laplacian.py`

**Interfaces:**
- Consumes: `networkx.Graph` from Task A2.
- Produces: `combinatorial_laplacian(G)`, `spectral_embedding(G, k=8)`, `spectral_gap(G, k=8)`.

- [ ] **Step 1: Write the failing test.** Two disjoint triangles → algebraic connectivity (2nd eigenvalue) ≈ 0, and the Fiedler vector separates the components.

```python
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
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_spectral_laplacian.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""Combinatorial graph Laplacian L = D - A and its low eigenvectors.

CPU via scipy.sparse.linalg.eigsh. A `cugraph` backend can replace the eig
call later without changing callers (see the `gpu` extra); keep the return
contract identical.
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


def _small_eigs(L: sp.csr_matrix, k: int) -> np.ndarray:
    n = L.shape[0]
    k_eff = min(k, n - 1)
    # shift-invert near 0 for the smallest eigenvalues; dense fallback for tiny graphs
    if n <= 12:
        w = np.linalg.eigvalsh(L.toarray())
        return np.sort(w)[: k + 1]
    vals = spla.eigsh(L, k=k_eff + 1, sigma=0, which="LM", return_eigenvectors=False)
    return np.sort(vals)


def spectral_gap(G: nx.Graph, k: int = 8) -> list[float]:
    _, L = combinatorial_laplacian(G)
    return [float(v) for v in _small_eigs(L, k)]


def spectral_embedding(G: nx.Graph, k: int = 8) -> tuple[list[str], np.ndarray]:
    nodes, L = combinatorial_laplacian(G)
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
```

- [ ] **Step 4: Run to verify it passes.** Run: `pytest tests/test_spectral_laplacian.py -q` — Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/agent_graph/spectral/laplacian.py tests/test_spectral_laplacian.py
git commit -m "feat(spectral): Laplacian, spectral embedding, spectral gap"
```

---

### Task A4: community detection (Louvain default, Leiden optional)

**Files:**
- Create: `src/agent_graph/spectral/communities.py`
- Test: `tests/test_spectral_communities.py`

**Interfaces:**
- Produces: `detect_communities(G, method="louvain", resolution=1.0, seed=0) -> dict[str, int]`

- [ ] **Step 1: Write the failing test.** Two triangles bridged by one weak edge → two communities.

```python
import networkx as nx
from agent_graph.spectral.communities import detect_communities

def test_louvain_finds_two_communities():
    G = nx.Graph()
    for a, b in [("a","b"),("b","c"),("a","c"),("x","y"),("y","z"),("x","z")]:
        G.add_edge(a, b, weight=1.0)
    G.add_edge("c", "x", weight=0.1)                 # weak bridge
    labels = detect_communities(G, method="louvain", seed=0)
    assert len({labels[n] for n in G.nodes()}) == 2
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["x"] == labels["y"] == labels["z"]
    assert labels["a"] != labels["x"]
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_spectral_communities.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** (Louvain via networkx built-in; Leiden lazy-imported only if requested).

```python
"""Community detection on the entity graph. Louvain (networkx built-in, no
extra deps) is the default; Leiden is available via the `leiden` extra and is
lazy-imported so CPU-only installs don't require igraph/leidenalg."""
from __future__ import annotations

import networkx as nx


def detect_communities(
    G: nx.Graph, method: str = "louvain", resolution: float = 1.0, seed: int = 0
) -> dict[str, int]:
    if method == "louvain":
        comms = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=seed
        )
    elif method == "leiden":
        comms = _leiden(G, resolution=resolution, seed=seed)
    else:
        raise ValueError(f"unknown method: {method!r} (use 'louvain' or 'leiden')")
    return {node: cid for cid, community in enumerate(comms) for node in community}


def _leiden(G: nx.Graph, resolution: float, seed: int):
    import igraph as ig            # lazy: only needed for method="leiden"
    import leidenalg as la
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges)
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition, weights=weights,
        resolution_parameter=resolution, seed=seed,
    )
    return [frozenset(nodes[i] for i in group) for group in part]
```

- [ ] **Step 4: Run to verify it passes.** Run: `pytest tests/test_spectral_communities.py -q` — Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/agent_graph/spectral/communities.py tests/test_spectral_communities.py
git commit -m "feat(spectral): Louvain (default) + optional Leiden communities"
```

---

### Task A5: bridging centrality

**Files:**
- Create: `src/agent_graph/spectral/centrality.py`
- Test: `tests/test_spectral_centrality.py`

**Interfaces:**
- Produces: `bridging_centrality(G) -> dict[str, float]`

- [ ] **Step 1: Write the failing test.** In two triangles joined by a single bridge node path, the bridge endpoints score highest.

```python
import networkx as nx
from agent_graph.spectral.centrality import bridging_centrality

def test_bridge_nodes_score_highest():
    G = nx.Graph()
    for a, b in [("a","b"),("b","c"),("a","c"),("x","y"),("y","z"),("x","z")]:
        G.add_edge(a, b, weight=1.0)
    G.add_edge("c", "x", weight=1.0)                 # c and x are the bridge
    bc = bridging_centrality(G)
    assert bc["c"] > bc["a"] and bc["x"] > bc["z"]
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_spectral_centrality.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** (bridging centrality = betweenness × bridging coefficient, Hwang et al.).

```python
"""Bridging centrality: betweenness weighted by the bridging coefficient, which
up-weights nodes that sit between densely-connected regions (structural holes).
bridging_coeff(v) = (1/deg(v)) / sum_{u in N(v)} 1/deg(u)."""
from __future__ import annotations

import networkx as nx


def bridging_centrality(G: nx.Graph) -> dict[str, float]:
    betw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    deg = dict(G.degree())
    out: dict[str, float] = {}
    for v in G.nodes():
        neigh = list(G.neighbors(v))
        if not neigh or deg[v] == 0:
            out[v] = 0.0
            continue
        denom = sum(1.0 / deg[u] for u in neigh if deg[u] > 0)
        coeff = (1.0 / deg[v]) / denom if denom > 0 else 0.0
        out[v] = betw[v] * coeff
    return out
```

- [ ] **Step 4: Run to verify it passes.** Run: `pytest tests/test_spectral_centrality.py -q` — Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/agent_graph/spectral/centrality.py tests/test_spectral_centrality.py
git commit -m "feat(spectral): bridging centrality"
```

---

### Task A6: versioned spectral snapshots as first-class KG nodes

**Files:**
- Create: `src/agent_graph/spectral/snapshots.py`
- Test: `tests/test_spectral_snapshots.py`

**Interfaces:**
- Consumes: a `pyoxigraph.Store`; outputs of A3/A4/A5.
- Produces: `SpectralSnapshot` dataclass; `write_spectral_snapshot(store, snapshot, *, graph_name=None) -> str`.

This is Scier's "maintain versioned spectral snapshots as first-class KG nodes" bullet, made literal: a snapshot is a KG node whose annotations are the spectral gap, community assignment, and a content hash + timestamp for versioning.

- [ ] **Step 1: Write the failing test.** Compute a snapshot, write it, and assert the snapshot node + its `spectral_gap`/`computed_at` triples are queryable back out.

```python
import pyoxigraph as ox
import networkx as nx
from agent_graph.spectral.snapshots import SpectralSnapshot, write_spectral_snapshot
from agent_graph.spectral.laplacian import spectral_gap
from agent_graph.spectral.communities import detect_communities

def _two_triangles_store():
    G = nx.Graph()
    for a, b in [("a","b"),("b","c"),("a","c"),("x","y"),("y","z"),("x","z")]:
        G.add_edge(a, b, weight=1.0)
    return G

def test_snapshot_is_written_and_queryable():
    store = ox.Store()
    G = _two_triangles_store()
    snap = SpectralSnapshot(
        gap=spectral_gap(G, k=3),
        communities=detect_communities(G, seed=0),
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
    uri = write_spectral_snapshot(store, snap)
    assert uri.startswith("http")
    # the snapshot node carries a spectral_gap[0] annotation
    SG = "http://ex/spectral/gap0"
    got = list(store.quads_for_pattern(ox.NamedNode(uri), ox.NamedNode(SG), None, None))
    assert len(got) == 1
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_spectral_snapshots.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""Persist a versioned spectral snapshot as first-class KG nodes.

A snapshot node (IRI keyed by content hash) carries the spectral gap, node/edge
counts, timestamp, and one membership triple per entity. Versioning: the IRI's
hash changes iff the graph or its spectral summary changes, so snapshots are
immutable and comparable across retraining cycles (Scier's spectral-snapshot
requirement)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pyoxigraph as ox

_NS = "http://ex/spectral/"
_GAP0 = ox.NamedNode(_NS + "gap0")
_NNODES = ox.NamedNode(_NS + "nNodes")
_NEDGES = ox.NamedNode(_NS + "nEdges")
_AT = ox.NamedNode(_NS + "computedAt")
_MEMBER = ox.NamedNode(_NS + "community")
_XSD_INT = ox.NamedNode("http://www.w3.org/2001/XMLSchema#integer")
_XSD_DBL = ox.NamedNode("http://www.w3.org/2001/XMLSchema#double")


@dataclass
class SpectralSnapshot:
    gap: list[float]
    communities: dict[str, int]
    n_nodes: int
    n_edges: int
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def content_hash(self) -> str:
        payload = json.dumps(
            {"gap": [round(g, 6) for g in self.gap],
             "comm": sorted(self.communities.items()),
             "n": [self.n_nodes, self.n_edges]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_spectral_snapshot(
    store: ox.Store, snapshot: SpectralSnapshot, *, graph_name=None
) -> str:
    uri = f"{_NS}snapshot/{snapshot.content_hash()}"
    node = ox.NamedNode(uri)
    g = (graph_name,) if graph_name is not None else ()

    def add(p, lit):
        store.add(ox.Quad(node, p, lit, *g))

    add(_GAP0, ox.Literal(str(snapshot.gap[0]), datatype=_XSD_DBL))
    add(_NNODES, ox.Literal(str(snapshot.n_nodes), datatype=_XSD_INT))
    add(_NEDGES, ox.Literal(str(snapshot.n_edges), datatype=_XSD_INT))
    add(_AT, ox.Literal(snapshot.computed_at))
    for entity, cid in snapshot.communities.items():
        store.add(ox.Quad(
            ox.NamedNode(entity), _MEMBER,
            ox.Literal(str(cid), datatype=_XSD_INT), *g,
        ))
    return uri
```

- [ ] **Step 4: Run to verify it passes.** Run: `pytest tests/test_spectral_snapshots.py -q` — Expected: PASS.
- [ ] **Step 5: Run the whole spectral suite + commit.**

```bash
pytest tests/test_spectral_*.py -q      # Expected: all PASS
git add src/agent_graph/spectral/snapshots.py tests/test_spectral_snapshots.py
git commit -m "feat(spectral): versioned spectral snapshots as KG nodes"
```

---

### Task A7: end-to-end demo + README section

**Files:**
- Create: `run_spectral_demo.py` (repo root, mirrors existing `run_demo.py`)
- Modify: `README.md` (add a "Spectral layer" section)

- [ ] **Step 1: Write the demo** — build a small claim graph in a pyoxigraph Store (or load the KG if present), run adapter → gap → communities → bridging → snapshot, and print a report. (No test; it's a script — the underlying functions are all tested.)

```python
"""python run_spectral_demo.py — spectral layer over a toy claim graph."""
import pyoxigraph as ox
from agent_graph.spectral import (
    build_entity_graph, spectral_gap, detect_communities,
    bridging_centrality, SpectralSnapshot, write_spectral_snapshot,
)

def _seed_store():
    store = ox.Store(); E = "http://ex/e/"; P = "http://ex/claim/affects"
    edges = [("A","B"),("B","C"),("A","C"),("X","Y"),("Y","Z"),("X","Z"),("C","X")]
    for s, o in edges:
        store.add(ox.Quad(ox.NamedNode(E+s), ox.NamedNode(P), ox.NamedNode(E+o)))
    return store

def main():
    store = _seed_store()
    G = build_entity_graph(store)
    print("gap:", [round(v, 4) for v in spectral_gap(G, k=3)])
    comm = detect_communities(G, seed=0)
    print("communities:", comm)
    bc = bridging_centrality(G)
    print("top bridge:", max(bc, key=bc.get))
    uri = write_spectral_snapshot(
        store, SpectralSnapshot(spectral_gap(G, 3), comm, G.number_of_nodes(), G.number_of_edges()))
    print("snapshot:", uri)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.** Run: `python run_spectral_demo.py` — Expected: prints gap (one near-zero eigenvalue for the bridged graph), a 2-community split, `C` or `X` as top bridge, and a snapshot IRI.
- [ ] **Step 3: README section** — add a short "Spectral layer" subsection documenting the public API and the demo command.
- [ ] **Step 4: Commit.**

```bash
git add run_spectral_demo.py README.md
git commit -m "docs(spectral): end-to-end demo + README section"
```

---

## Subsystems B / C / D — outlines (each gets its own plan when reached)

### B — Zero-flow CI edge validation (`src/agent_graph/kg/verify/`)
Attach data-driven confidence reifiers to causal claim edges via the zero-flow criterion (Wang et al. 2026, arXiv:2602.00797; causal_bench#85/SGA#8). Consumes: the KG reifier write-path + a per-edge `(X, Y, Z)` data matrix. Produces: `verify_edge(store, edge_iri, data) -> Verdict` writing a `supports|refutes|underpowered` annotation. **Own plan** — depends on the KG reifier API and a dataset; the rectified-flow CI estimator itself is the causal_bench#85 deliverable (do not duplicate; import or port).

### C — QLoRA fine-tune (`src/agent_graph/finetune/`)
Graph-aware supervised fine-tune of a small base model (start ≤7B, QLoRA) on KG-structured claim → summary pairs, so the model internalises the ontology structure. Tech: `peft`/`transformers`/`bitsandbytes`, single GPU, quantization-aware. Produces: a LoRA adapter + an eval hook into the existing assurance harness (faithfulness/consistency before vs after). **Own plan** — GPU-gated; this is the Mount Sinai + Scier fine-tuning artifact and the portfolio's single clean GPU demonstration.

**Roofline note — training vs serving are opposite regimes (do not conflate):**
- **QLoRA training is compute + VRAM bound**, not decode-bound: teacher-forced full-sequence forward+backward is dominated by the matmuls, and the binding constraint is VRAM (hence 4-bit quantization + gradient checkpointing), not memory bandwidth. The "one token at a time across big batches" concern does **not** apply here.
- **The memory-bandwidth decode regime is a *serving/eval* phenomenon** (autoregressive generation reloads weights + KV-cache per token → low arithmetic intensity → memory-BW-bound). It bites in **Subsystem B's eval harness Consistency@k** (generates k samples) and **Subsystem D's Gradio/RAG serving** — batch generations across requests/samples to amortise the weight load. Wire the batching there, not in the fine-tune loop. (Reference: the JAX scaling-book roofline/gpus chapters — TPU-centric but the arithmetic-intensity framing transfers to any accelerator.)

### D — Gradio MVP (`src/agent_graph/app/`)
A Gradio app: enter a query → run the existing retrieval/agent → render the claim subgraph, its communities (A4), and the current spectral snapshot (A6). Tech: `gradio`. Produces: `app.py` launchable with `python -m agent_graph.app`. **Own plan** — depends on A; the "rapid validation MVP" both Scier and Mount Sinai call for.

---

## Self-Review

**Spec coverage:** Program overview maps every net-new subsystem (A–D) to a role and a dependency; A is fully tasked (A1–A7 cover deps, adapter, Laplacian/embedding/gap, communities, centrality, snapshots, demo). B/C/D are explicitly deferred to their own plans per the Scope Check. Prerequisite (KG plan) is named. ✅

**Placeholder scan:** No TBD/"handle appropriately" — every code step carries real code; the one forward-reference (reifier-confidence weighting) is explicitly deferred to Task B with the interim behaviour (multiplicity weight) specified. ✅

**Type consistency:** `build_entity_graph → nx.Graph` consumed by `combinatorial_laplacian`/`detect_communities`/`bridging_centrality`; `SpectralSnapshot(gap, communities, n_nodes, n_edges)` fields match `write_spectral_snapshot` reads; `spectral_gap` returns `list[float]` used as `snap.gap[0]`. Names consistent across tasks and the `__init__` exports. ✅
