# Domain Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive RDF-star domain knowledge graph that extracts ontology-constrained scientific claims from paper abstracts, accumulates cross-paper Beta-Bernoulli confidence, flags (not resolves) contradictions, and injects weighted structured context into the summarizer.

**Architecture:** A new `src/agent_graph/kg/` package behind a `KnowledgeGraph` Protocol. Claims are stored in a pyoxigraph Store using the W3C RDF 1.2 reifier model (`rdf:reifies` + a reifier node carrying confidence/evidence/provenance annotations). A new `graph_builder_node` runs after the researcher and before the summarizer; the summarizer injects a weighted subgraph into its prompt. The flat-paper pipeline is untouched.

**Tech Stack:** Python 3.9+, pyoxigraph (RDF-star store, verified 0.5.9), Pydantic v2, langchain-anthropic, LangGraph, pytest.

## Global Constraints

- Python floor: `requires-python = ">=3.9"` — use `from typing import ...` with `typing_extensions` fallback for `NotRequired` (mirror `state.py`).
- Package layout is `src` layout (`package-dir = {"" = "src"}`); import as `agent_graph.kg.*`.
- LLM access goes through `agent_graph.llm.get_llm(temperature=..., max_tokens=...)` only — never construct `ChatAnthropic` directly.
- Tests live flat in `tests/` named `test_kg_*.py`, pytest function-style, imports inside test bodies (mirror `tests/test_schemas.py`). No network in any test.
- Run tests with `.venv/bin/pytest <path> -v` from repo root.
- Ontology is FROZEN (see Task 2). The extractor must be physically constrained to it via Pydantic `Literal`; no free-form predicates.
- RDF namespace constant: `KG = "https://kg.local/"`. Reifies IRI: `http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies`.
- Do NOT modify the uncommitted WIP in `run_demo.py` / `src/agent_graph/nodes.py:832` (pre-existing, unrelated).

---

### Task 1: Add pyoxigraph dependency

**Files:**
- Modify: `pyproject.toml` (dependencies array)

**Interfaces:**
- Produces: `pyoxigraph` importable as `import pyoxigraph as ox` in the venv.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` array (alphabetical near other libs):

```toml
    "pyoxigraph>=0.5",
```

- [ ] **Step 2: Install into the project venv**

Run: `VIRTUAL_ENV="$(pwd)/.venv" uv pip install "pyoxigraph>=0.5" -q`
Expected: completes without error.

- [ ] **Step 3: Verify import + RDF-star support**

Run: `.venv/bin/python -c "import pyoxigraph as ox; print(ox.RdfFormat.TURTLE.supports_rdf_star)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyoxigraph dependency for RDF-star knowledge graph"
```

---

### Task 2: Ontology module

**Files:**
- Create: `src/agent_graph/kg/__init__.py` (empty)
- Create: `src/agent_graph/kg/ontology.py`
- Test: `tests/test_kg_ontology.py`

**Interfaces:**
- Produces:
  - `ENTITY_TYPES: tuple[str,...]`, `EntityType` (Literal), `RELATION_TYPES: tuple[str,...]`, `RelationType` (Literal)
  - `FUNCTIONAL: set[str]` (empty), `MUTEX: set[tuple[str,str]]`
  - `KG: str` namespace; `RDF_REIFIES: str`
  - `slugify(name: str) -> str`
  - `entity_uri(etype: str, name: str) -> str`
  - `paper_uri(paper_id: str | None = None, pmid: str | None = None, arxiv_id: str | None = None) -> str`
  - `relation_uri(rel: str) -> str`
  - `mutex_partners(rel: str) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_ontology.py
"""Tests for the frozen KG ontology and URI minting."""
import pytest


def test_slugify_normalizes():
    from agent_graph.kg.ontology import slugify
    assert slugify("  Imatinib Mesylate ") == "imatinib-mesylate"
    assert slugify("Non-Small_Cell") == "non-small-cell"
    assert slugify("CD19+") == "cd19"


def test_entity_uri():
    from agent_graph.kg.ontology import entity_uri
    assert entity_uri("drug", "Imatinib") == "https://kg.local/drug/imatinib"


def test_paper_uri_precedence():
    from agent_graph.kg.ontology import paper_uri
    assert paper_uri(pmid="12345") == "pmid:12345"
    assert paper_uri(arxiv_id="2401.001") == "arxiv:2401.001"
    assert paper_uri(paper_id="Some Local ID") == "paper:some-local-id"


def test_mutex_is_symmetric_and_partners():
    from agent_graph.kg.ontology import mutex_partners
    assert "decreases_risk_of" in mutex_partners("increases_risk_of")
    assert "increases_risk_of" in mutex_partners("decreases_risk_of")
    assert mutex_partners("treats") == []


def test_functional_is_empty_but_present():
    from agent_graph.kg import ontology
    assert ontology.FUNCTIONAL == set()


def test_ontology_frozen_membership():
    from agent_graph.kg.ontology import ENTITY_TYPES, RELATION_TYPES
    assert set(ENTITY_TYPES) == {
        "drug", "disease", "gene", "biomarker", "method", "population", "paper", "author"
    }
    for rel in ("treats", "increases_risk_of", "subtype_of", "wrote"):
        assert rel in RELATION_TYPES
    assert "affiliated_with" not in RELATION_TYPES  # reserved
    assert "cites" not in RELATION_TYPES            # reserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_ontology.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_graph.kg'`

- [ ] **Step 3: Create the package and module**

```python
# src/agent_graph/kg/__init__.py
```
(empty file)

```python
# src/agent_graph/kg/ontology.py
"""Frozen ontology for the scientific-literature knowledge graph.

Declared ONCE here. The extractor (extract.py) is constrained to these via
Pydantic Literals, so the LLM cannot mint off-ontology predicates. This is the
single anti-sprawl gate.
"""
import re
from typing import Literal, get_args

KG = "https://kg.local/"
RDF_REIFIES = "http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies"

EntityType = Literal[
    "drug", "disease", "gene", "biomarker", "method", "population", "paper", "author"
]
ENTITY_TYPES = get_args(EntityType)

# Claim predicates + bibliographic `wrote`. `cites` / `affiliated_with` are RESERVED.
RelationType = Literal[
    "treats", "causes", "associated_with", "inhibits", "increases_risk_of",
    "decreases_risk_of", "measured_by", "subtype_of", "studied_in", "wrote",
]
RELATION_TYPES = get_args(RelationType)

# FUNCTIONAL is intentionally empty (nothing is single-valued in a literature graph)
# but the mechanism is retained for the clinical extension. MUTEX flags genuine
# logical contradiction (flag-not-reject).
FUNCTIONAL: set = set()
MUTEX: set = {("increases_risk_of", "decreases_risk_of")}


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)      # drop punctuation
    s = re.sub(r"[\s_]+", "-", s)        # spaces/underscores -> hyphen
    return s.strip("-")


def entity_uri(etype: str, name: str) -> str:
    return f"{KG}{etype}/{slugify(name)}"


def paper_uri(paper_id: str = None, pmid: str = None, arxiv_id: str = None) -> str:
    if pmid:
        return f"pmid:{pmid}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"paper:{slugify(paper_id or 'unknown')}"


def relation_uri(rel: str) -> str:
    return f"{KG}{rel}"


def mutex_partners(rel: str) -> list:
    out = []
    for a, b in MUTEX:
        if rel == a:
            out.append(b)
        elif rel == b:
            out.append(a)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_ontology.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/kg/__init__.py src/agent_graph/kg/ontology.py tests/test_kg_ontology.py
git commit -m "feat(kg): frozen ontology + URI minting"
```

---

### Task 3: Beta-Bernoulli confidence module

**Files:**
- Create: `src/agent_graph/kg/confidence.py`
- Test: `tests/test_kg_confidence.py`

**Interfaces:**
- Produces:
  - `PRIOR_A: float`, `PRIOR_B: float`
  - `evidence_weight(relevance: float) -> float`
  - `beta_params(evidence: list[dict], prior_a: float = PRIOR_A, prior_b: float = PRIOR_B) -> tuple[float, float]`
  - `confidence(alpha: float, beta: float) -> float`
  - `confidence_lb(alpha: float, beta: float, z: float = 1.64) -> float`
- Evidence dict shape consumed: `{"relevance": int 1-100, "polarity": "supports"|"refutes", ...}`. Missing `relevance` defaults to 50; missing `polarity` defaults to `"supports"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_confidence.py
"""Tests for Beta-Bernoulli confidence accumulation."""
import pytest


def test_more_support_raises_confidence():
    from agent_graph.kg.confidence import beta_params, confidence
    one = [{"relevance": 80, "polarity": "supports"}]
    three = one * 3
    assert confidence(*beta_params(three)) > confidence(*beta_params(one))


def test_refutation_lowers_confidence():
    from agent_graph.kg.confidence import beta_params, confidence
    supp = [{"relevance": 80, "polarity": "supports"}]
    mixed = supp + [{"relevance": 80, "polarity": "refutes"}] * 3
    assert confidence(*beta_params(mixed)) < confidence(*beta_params(supp))


def test_thin_evidence_penalized_by_lower_bound():
    from agent_graph.kg.confidence import beta_params, confidence, confidence_lb
    thin = [{"relevance": 90, "polarity": "supports"}]
    thick = [{"relevance": 60, "polarity": "supports"}] * 8
    # thick may have similar/lower mean but a higher lower-bound (less uncertainty)
    assert confidence_lb(*beta_params(thick)) > confidence_lb(*beta_params(thin))


def test_defaults_for_missing_fields():
    from agent_graph.kg.confidence import beta_params
    a, b = beta_params([{}])  # relevance->50, polarity->supports
    assert a > 1.0 and b == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_confidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_graph.kg.confidence'`

- [ ] **Step 3: Write the implementation**

```python
# src/agent_graph/kg/confidence.py
"""Beta-Bernoulli confidence for KG claims.

Each claim is a Beta(alpha, beta) belief. Supporting papers add to alpha,
refuting papers add to beta, weighted by relevance as fractional pseudo-counts.
Point estimate = posterior mean; ranking uses a credible-interval lower bound
so thin evidence is penalized.
"""
from math import sqrt

PRIOR_A = 1.0
PRIOR_B = 1.0


def evidence_weight(relevance: float) -> float:
    return 0.3 + 0.65 * (relevance / 100.0)


def beta_params(evidence: list, prior_a: float = PRIOR_A, prior_b: float = PRIOR_B):
    a = prior_a + sum(
        evidence_weight(e.get("relevance", 50))
        for e in evidence if e.get("polarity", "supports") == "supports"
    )
    b = prior_b + sum(
        evidence_weight(e.get("relevance", 50))
        for e in evidence if e.get("polarity") == "refutes"
    )
    return a, b


def confidence(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def confidence_lb(alpha: float, beta: float, z: float = 1.64) -> float:
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return max(0.0, mean - z * sqrt(var))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_confidence.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/kg/confidence.py tests/test_kg_confidence.py
git commit -m "feat(kg): Beta-Bernoulli confidence model"
```

---

### Task 4: Extraction schema + prompt

**Files:**
- Create: `src/agent_graph/kg/extract.py`
- Test: `tests/test_kg_extract.py`

**Interfaces:**
- Produces:
  - `ScientificTriplet` (Pydantic): fields `subject: str`, `subject_type: EntityType`, `relation: RelationType`, `object: str`, `object_type: EntityType`, `polarity: Literal["supports","refutes"] = "supports"`
  - `TripletExtraction` (Pydantic): `triplets: list[ScientificTriplet] = []`
  - `Evidence` (Pydantic): the typed boundary object for one paper's assertion of a claim — `paper_uri: str`, `pmid: Optional[str]` (constrained `^\d+$`), `paper_id: Optional[str]`, `pub_year: Optional[int]` (1800–2100), `relevance: int` (0–100), `polarity: Literal["supports","refutes"] = "supports"`, `snippet: str = ""`. This is the **parse-don't-validate** boundary type: `add_relation` (Task 5) coerces any incoming `dict` into `Evidence` once, so nothing downstream re-checks a raw blob.
  - `EXTRACTION_PROMPT: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_extract.py
"""Tests for the ontology-constrained extraction schema."""
import pytest
from pydantic import ValidationError


def test_valid_triplet():
    from agent_graph.kg.extract import ScientificTriplet
    t = ScientificTriplet(subject="Imatinib", subject_type="drug",
                          relation="treats", object="CML", object_type="disease")
    assert t.polarity == "supports"


def test_off_ontology_relation_rejected():
    from agent_graph.kg.extract import ScientificTriplet
    with pytest.raises(ValidationError):
        ScientificTriplet(subject="A", subject_type="drug",
                          relation="cites", object="B", object_type="paper")


def test_off_ontology_entity_type_rejected():
    from agent_graph.kg.extract import ScientificTriplet
    with pytest.raises(ValidationError):
        ScientificTriplet(subject="A", subject_type="organization",
                          relation="treats", object="B", object_type="disease")


def test_empty_extraction_default():
    from agent_graph.kg.extract import TripletExtraction
    assert TripletExtraction().triplets == []


def test_evidence_valid_and_constrained_pmid():
    from agent_graph.kg.extract import Evidence
    e = Evidence(paper_uri="pmid:12345", pmid="12345", relevance=90)
    assert e.pmid == "12345" and e.polarity == "supports"


def test_evidence_rejects_nonnumeric_pmid_and_bad_relevance():
    from agent_graph.kg.extract import Evidence
    with pytest.raises(ValidationError):
        Evidence(paper_uri="x", pmid="PMC-not-a-pmid", relevance=50)
    with pytest.raises(ValidationError):
        Evidence(paper_uri="x", relevance=500)  # out of 0-100


def test_evidence_coerces_from_dict():
    from agent_graph.kg.extract import Evidence
    e = Evidence.model_validate({"paper_uri": "arxiv:2401.1", "relevance": 70})
    assert e.pmid is None and e.relevance == 70
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_graph.kg.extract'`

- [ ] **Step 3: Write the implementation**

```python
# src/agent_graph/kg/extract.py
"""Ontology-constrained triplet extraction schema + prompt.

The Literals are imported from ontology.py so the schema and the gate cannot
drift apart. Pydantic enforces them at the structured-output boundary.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field

from agent_graph.kg.ontology import EntityType, RelationType


class ScientificTriplet(BaseModel):
    subject: str
    subject_type: EntityType
    relation: RelationType
    object: str
    object_type: EntityType
    polarity: Literal["supports", "refutes"] = "supports"


class TripletExtraction(BaseModel):
    triplets: list[ScientificTriplet] = Field(default_factory=list)


class Evidence(BaseModel):
    """One paper's assertion of a claim — the parse-don't-validate boundary type.

    add_relation() coerces incoming dicts into this once, so the constrained
    pmid / bounded relevance / known polarity are guaranteed for everything
    downstream (confidence, serialization). A raw dict never flows deeper.
    """
    paper_uri: str
    pmid: Optional[str] = Field(default=None, pattern=r"^\d+$")
    paper_id: Optional[str] = None
    pub_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    relevance: int = Field(ge=0, le=100)
    polarity: Literal["supports", "refutes"] = "supports"
    snippet: str = ""


EXTRACTION_PROMPT = """You extract structured scientific claims from a paper abstract.

Rules:
- Extract ONLY relationships expressible with the allowed entity types and relations.
  Anything that does not fit is dropped — do not invent relations.
- Allowed entity types: drug, disease, gene, biomarker, method, population, paper, author.
- Allowed relations: treats, causes, associated_with, inhibits, increases_risk_of,
  decreases_risk_of, measured_by, subtype_of, studied_in, wrote.
- Set polarity="refutes" when the paper reports a NULL or NEGATIVE finding for a claim
  (e.g. "no significant effect"); otherwise polarity="supports".
- Extract claims the paper makes; do not extract the question or background it cites.

Return the triplets for the abstract below."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_extract.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/kg/extract.py tests/test_kg_extract.py
git commit -m "feat(kg): ontology-constrained extraction schema + typed Evidence boundary"
```

---

### Task 5: Protocol + store — add_relation, recompute, conflict flagging

**Files:**
- Create: `src/agent_graph/kg/protocol.py`
- Create: `src/agent_graph/kg/rdfstar_store.py`
- Test: `tests/test_kg_store_write.py`

**Interfaces:**
- Consumes: `ontology` (Task 2), `confidence` (Task 3), `extract.Evidence` (Task 4).
- Produces:
  - `protocol.KnowledgeGraph` (Protocol) with: `add_relation(subject, subject_type, relation, obj, object_type, evidence: dict | Evidence) -> str | None`, `query(...) -> list[dict]`, `to_context(edges, limit=25) -> str`, `to_dict() -> dict`, classmethod `from_dict(data: dict)`.
  - `rdfstar_store.OxigraphKG` implementing it. This task delivers `__init__`, `add_relation`, and internal helpers `_node`, `_reifier_for`, `_evidence`, `_dedupe`, `_write_annotations`, `_flag_conflicts`. (`query`/`to_context`/`to_dict`/`from_dict`/`merge_from` arrive in Tasks 6–7.)
  - **Parse-don't-validate boundary:** `add_relation` coerces its `evidence` argument through `Evidence.model_validate(...).model_dump()` as the first line, so a malformed blob (bad pmid, out-of-range relevance) raises `ValidationError` *here*, and the internal dict-based machinery (`_dedupe`, `_write_annotations`, `beta_params`) only ever sees validated fields. Validated dict shape: `{"paper_uri": str, "pmid": str|None, "paper_id": str|None, "pub_year": int|None, "relevance": int, "polarity": "supports"|"refutes", "snippet": str}`. Dedup key is `paper_uri` (or `pmid`/`paper_id` fallback) per claim — re-adding the same paper to the same claim is idempotent.
  - Annotation predicate locals on the reifier: `confidence, confidence_lb, alpha, beta, support, refute, first_year, last_year, contested, evidence, asserted_by` (all under `KG` namespace; `asserted_by` objects are paper URIs).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_store_write.py
"""Tests for OxigraphKG writes: confidence accrual + contradiction flagging."""
import pytest


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity,
            "snippet": "t"}


def test_add_relation_creates_claim_with_confidence():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    meta = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")
    assert meta is not None
    assert 0.0 < meta["confidence"] <= 1.0
    assert meta["support"] == 1


def test_more_papers_raise_confidence_and_are_idempotent():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    c1 = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")["confidence"]
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("2"))
    c2 = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")["confidence"]
    assert c2 > c1
    # re-adding paper "2" must NOT increase support (idempotent per paper)
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("2"))
    meta = kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")
    assert meta["support"] == 2


def test_mutex_flags_both_claims_contested():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("StatinX", "drug", "increases_risk_of", "MI", "disease", _ev("1"))
    note = kg.add_relation("StatinX", "drug", "decreases_risk_of", "MI", "disease", _ev("2"))
    assert note is not None and "contradiction" in note
    m_inc = kg._claim_meta("drug", "StatinX", "increases_risk_of", "disease", "MI")
    m_dec = kg._claim_meta("drug", "StatinX", "decreases_risk_of", "disease", "MI")
    assert m_inc["contested"] is True and m_dec["contested"] is True


def test_add_relation_rejects_malformed_evidence_at_boundary():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    from pydantic import ValidationError
    kg = OxigraphKG()
    with pytest.raises(ValidationError):
        kg.add_relation("Imatinib", "drug", "treats", "CML", "disease",
                        {"paper_uri": "x", "pmid": "not-numeric", "relevance": 50})
    # nothing was written — the blob never reached the store
    assert kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_store_write.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_graph.kg.rdfstar_store'`

- [ ] **Step 3: Write the Protocol**

```python
# src/agent_graph/kg/protocol.py
"""Backend-agnostic KnowledgeGraph interface. Nodes import only this."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class KnowledgeGraph(Protocol):
    def add_relation(self, subject: str, subject_type: str, relation: str,
                     obj: str, object_type: str, evidence):
        """Upsert a claim, fold in one evidence item, recompute confidence,
        flag MUTEX conflicts. Returns a conflict note string or None.

        `evidence` is a dict or extract.Evidence; the implementation parses it
        into a validated Evidence at the boundary (raises ValidationError on a
        malformed blob)."""
        ...

    def query(self, entities, relation_hints=None, max_depth: int = 2,
              min_confidence: float = 0.0, as_of_year: int = None) -> list:
        ...

    def to_context(self, edges: list, limit: int = 25) -> str:
        ...

    def to_dict(self) -> dict:
        ...
```

- [ ] **Step 4: Write the store (write path only)**

```python
# src/agent_graph/kg/rdfstar_store.py
"""pyoxigraph implementation of KnowledgeGraph using the RDF 1.2 reifier model.

Each claim = a base triple (s, p, o) plus a reifier blank node r with
(r, rdf:reifies, <<s p o>>). Aggregate stats and the evidence JSON hang on r.
"""
import io
import json
import pyoxigraph as ox

from agent_graph.kg import ontology as ont
from agent_graph.kg import confidence as conf
from agent_graph.kg.extract import Evidence

_REIFIES = ox.NamedNode(ont.RDF_REIFIES)
_XSD_DOUBLE = ox.NamedNode("http://www.w3.org/2001/XMLSchema#double")
_XSD_INT = ox.NamedNode("http://www.w3.org/2001/XMLSchema#integer")
_XSD_BOOL = ox.NamedNode("http://www.w3.org/2001/XMLSchema#boolean")


def _pred(local):
    return ox.NamedNode(ont.KG + local)


# annotation predicates
_EVIDENCE = _pred("evidence")
_CONF = _pred("confidence")
_CONF_LB = _pred("confidence_lb")
_ALPHA = _pred("alpha")
_BETA = _pred("beta")
_SUPPORT = _pred("support")
_REFUTE = _pred("refute")
_FIRST_YEAR = _pred("first_year")
_LAST_YEAR = _pred("last_year")
_CONTESTED = _pred("contested")
_ASSERTED_BY = _pred("asserted_by")

# predicates that are real graph edges (vs reifier metadata)
_EDGE_PREDICATES = set(ont.RELATION_TYPES)


class OxigraphKG:
    def __init__(self, store=None):
        self.store = store if store is not None else ox.Store()

    # ---- node construction ----
    def _node(self, etype: str, name: str) -> "ox.NamedNode":
        if etype == "paper":
            if name.startswith(("pmid:", "arxiv:", "paper:")):
                return ox.NamedNode(name)
            return ox.NamedNode(ont.paper_uri(paper_id=name))
        return ox.NamedNode(ont.entity_uri(etype, name))

    def _claim_triple(self, s_type, s_name, relation, o_type, o_name) -> "ox.Triple":
        return ox.Triple(self._node(s_type, s_name),
                         ox.NamedNode(ont.relation_uri(relation)),
                         self._node(o_type, o_name))

    def _reifier_for(self, claim: "ox.Triple"):
        for q in self.store.quads_for_pattern(None, _REIFIES, claim, None):
            return q.subject
        return None

    # ---- annotation read/write ----
    def _set(self, r, pred, node):
        for q in list(self.store.quads_for_pattern(r, pred, None, None)):
            self.store.remove(q)
        self.store.add(ox.Quad(r, pred, node))

    def _evidence(self, r) -> list:
        for q in self.store.quads_for_pattern(r, _EVIDENCE, None, None):
            return json.loads(q.object.value)
        return []

    @staticmethod
    def _ev_key(e: dict) -> str:
        return e.get("paper_uri") or e.get("pmid") or e.get("paper_id") or ""

    def _dedupe(self, evidence: list) -> list:
        seen, out = set(), []
        for e in evidence:
            k = self._ev_key(e)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out

    def _write_annotations(self, r, evidence: list):
        evidence = self._dedupe(evidence)
        a, b = conf.beta_params(evidence)
        self._set(r, _EVIDENCE, ox.Literal(json.dumps(evidence)))
        self._set(r, _CONF, ox.Literal(repr(conf.confidence(a, b)), datatype=_XSD_DOUBLE))
        self._set(r, _CONF_LB, ox.Literal(repr(conf.confidence_lb(a, b)), datatype=_XSD_DOUBLE))
        self._set(r, _ALPHA, ox.Literal(repr(a), datatype=_XSD_DOUBLE))
        self._set(r, _BETA, ox.Literal(repr(b), datatype=_XSD_DOUBLE))
        sup = sum(1 for e in evidence if e.get("polarity", "supports") == "supports")
        ref = sum(1 for e in evidence if e.get("polarity") == "refutes")
        self._set(r, _SUPPORT, ox.Literal(str(sup), datatype=_XSD_INT))
        self._set(r, _REFUTE, ox.Literal(str(ref), datatype=_XSD_INT))
        years = [e["pub_year"] for e in evidence if e.get("pub_year")]
        if years:
            self._set(r, _FIRST_YEAR, ox.Literal(str(min(years)), datatype=_XSD_INT))
            self._set(r, _LAST_YEAR, ox.Literal(str(max(years)), datatype=_XSD_INT))
        # asserted_by provenance links (additive, deduped by URI)
        have = {q.object.value for q in self.store.quads_for_pattern(r, _ASSERTED_BY, None, None)}
        for e in evidence:
            uri = e.get("paper_uri")
            if uri and uri not in have:
                self.store.add(ox.Quad(r, _ASSERTED_BY, ox.NamedNode(uri)))
                have.add(uri)

    # ---- FUNCTIONAL is empty; mechanism retained for the clinical extension ----
    def _functional_violation(self, relation, s_node):
        if relation not in ont.FUNCTIONAL:
            return None
        objs = {q.object.value for q in self.store.quads_for_pattern(s_node, ox.NamedNode(ont.relation_uri(relation)), None, None)}
        return f"functional-violation: {s_node.value} {relation} {sorted(objs)}" if len(objs) > 1 else None

    def _flag_conflicts(self, relation, s_node, o_node):
        for other in ont.mutex_partners(relation):
            partner = ox.Triple(s_node, ox.NamedNode(ont.relation_uri(other)), o_node)
            r2 = self._reifier_for(partner)
            if r2 is not None:
                this = ox.Triple(s_node, ox.NamedNode(ont.relation_uri(relation)), o_node)
                r1 = self._reifier_for(this)
                self._set(r1, _CONTESTED, ox.Literal("true", datatype=_XSD_BOOL))
                self._set(r2, _CONTESTED, ox.Literal("true", datatype=_XSD_BOOL))
                return (f"contradiction: {s_node.value} {relation}/{other} "
                        f"{o_node.value} (both retained)")
        return self._functional_violation(relation, s_node)

    # ---- public write ----
    def add_relation(self, subject, subject_type, relation, obj, object_type, evidence):
        # parse-don't-validate boundary: coerce the blob into a typed Evidence ONCE.
        # Malformed input (bad pmid / out-of-range relevance) raises here, before
        # anything touches the store; downstream only sees validated fields.
        evidence = Evidence.model_validate(evidence).model_dump()
        s = self._node(subject_type, subject)
        o = self._node(object_type, obj)
        p = ox.NamedNode(ont.relation_uri(relation))
        claim = ox.Triple(s, p, o)
        self.store.add(ox.Quad(s, p, o))  # base fact (idempotent in oxigraph)
        r = self._reifier_for(claim)
        if r is None:
            r = ox.BlankNode()
            self.store.add(ox.Quad(r, _REIFIES, claim))
            ev_list = []
        else:
            ev_list = self._evidence(r)
        ev_list.append(evidence)
        self._write_annotations(r, ev_list)
        return self._flag_conflicts(relation, s, o)

    # ---- test/inspection helper (also used by query in Task 6) ----
    def _claim_meta(self, s_type, s_name, relation, o_type, o_name):
        claim = self._claim_triple(s_type, s_name, relation, o_type, o_name)
        r = self._reifier_for(claim)
        if r is None:
            return None
        return self._reifier_meta(r)

    def _reifier_meta(self, r) -> dict:
        m = {"confidence": 0.0, "confidence_lb": 0.0, "support": 0, "refute": 0,
             "first_year": None, "last_year": None, "contested": False}
        for q in self.store.quads_for_pattern(r, None, None, None):
            local = q.predicate.value.replace(ont.KG, "")
            v = getattr(q.object, "value", None)
            if local in ("confidence", "confidence_lb", "alpha", "beta"):
                m[local] = float(v)
            elif local in ("support", "refute", "first_year", "last_year"):
                m[local] = int(v)
            elif local == "contested":
                m[local] = (v == "true")
        return m
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_store_write.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/agent_graph/kg/protocol.py src/agent_graph/kg/rdfstar_store.py tests/test_kg_store_write.py
git commit -m "feat(kg): oxigraph store write path + RDF-star conflict flagging"
```

---

### Task 6: Store — weighted BFS query + to_context

**Files:**
- Modify: `src/agent_graph/kg/rdfstar_store.py` (add `query`, `to_context`)
- Test: `tests/test_kg_store_query.py`

**Interfaces:**
- Consumes: write path + `_reifier_meta` (Task 5).
- Produces:
  - `query(entities, relation_hints=None, max_depth=2, min_confidence=0.0, as_of_year=None) -> list[dict]` where each dict is `{"subject","relation","object","confidence","confidence_lb","support","contested","years":(first,last),"hint_match"}`, ranked by `(hint_match, confidence_lb, last_year)` descending.
  - `to_context(edges, limit=25) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_store_query.py
"""Tests for weighted BFS traversal + context serialization."""
import pytest


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity, "snippet": "t"}


def _graph():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    kg.add_relation("CML", "disease", "subtype_of", "Leukemia", "disease", _ev("2", year=2010))
    kg.add_relation("Aspirin", "drug", "treats", "Headache", "disease", _ev("3", relevance=30))
    return kg


def test_query_finds_seed_and_neighbors():
    kg = _graph()
    edges = kg.query(["Imatinib"], max_depth=2)
    rels = {(e["subject"].split("/")[-1], e["relation"], e["object"].split("/")[-1]) for e in edges}
    assert ("imatinib", "treats", "cml") in rels
    assert ("cml", "subtype_of", "leukemia") in rels       # reached at depth 2
    assert all("aspirin" not in e["subject"] for e in edges)  # disconnected


def test_min_confidence_filter():
    kg = _graph()
    hi = kg.query(["Aspirin"], min_confidence=0.0)
    lo = kg.query(["Aspirin"], min_confidence=0.95)
    assert len(hi) >= 1 and len(lo) == 0


def test_as_of_year_filter():
    kg = _graph()
    edges = kg.query(["CML"], max_depth=1, as_of_year=2005)
    # subtype_of (2010) is excluded; treats-CML (2020) excluded; nothing <= 2005
    assert all(e["years"][0] is None or e["years"][0] <= 2005 for e in edges)


def test_hint_match_ranks_first():
    kg = _graph()
    edges = kg.query(["CML"], relation_hints=["subtype_of"], max_depth=1)
    assert edges[0]["relation"] == "subtype_of"


def test_to_context_marks_contested():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("StatinX", "drug", "increases_risk_of", "MI", "disease", _ev("1"))
    kg.add_relation("StatinX", "drug", "decreases_risk_of", "MI", "disease", _ev("2"))
    edges = kg.query(["StatinX"], max_depth=1)
    ctx = kg.to_context(edges)
    assert "CONTESTED" in ctx and "increases_risk_of" in ctx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_store_query.py -v`
Expected: FAIL — `AttributeError: 'OxigraphKG' object has no attribute 'query'`

- [ ] **Step 3: Add query + to_context to the store**

Append these methods to `class OxigraphKG` in `src/agent_graph/kg/rdfstar_store.py`:

```python
    def _exists(self, node) -> bool:
        for q in self.store.quads_for_pattern(node, None, None, None):
            if q.predicate.value.replace(ont.KG, "") in _EDGE_PREDICATES:
                return True
        for q in self.store.quads_for_pattern(None, None, node, None):
            if q.predicate.value.replace(ont.KG, "") in _EDGE_PREDICATES:
                return True
        return False

    def _seed_nodes(self, entities):
        seeds = []
        for name in entities:
            for et in ont.ENTITY_TYPES:
                node = self._node(et, name)
                if self._exists(node):
                    seeds.append(node)
        return seeds

    def query(self, entities, relation_hints=None, max_depth=2,
              min_confidence=0.0, as_of_year=None):
        hints = {h.lower().strip() for h in (relation_hints or [])}
        frontier = [(n, 0) for n in self._seed_nodes(entities)]
        seen_nodes = {n.value for n, _ in frontier}
        seen_edges, out = set(), []
        i = 0
        while i < len(frontier):
            node, depth = frontier[i]
            i += 1
            if depth >= max_depth:
                continue
            candidates = []
            for q in self.store.quads_for_pattern(node, None, None, None):
                candidates.append((q.subject, q.predicate, q.object))
            for q in self.store.quads_for_pattern(None, None, node, None):
                candidates.append((q.subject, q.predicate, q.object))
            for s, p, o in candidates:
                rel = p.value.replace(ont.KG, "")
                if rel not in _EDGE_PREDICATES:
                    continue
                ekey = (s.value, rel, o.value)
                if ekey in seen_edges:
                    continue
                r = self._reifier_for(ox.Triple(s, p, o))
                meta = self._reifier_meta(r) if r is not None else None
                if meta is None or meta["confidence"] < min_confidence:
                    continue
                if as_of_year is not None and meta["first_year"] and meta["first_year"] > as_of_year:
                    continue
                seen_edges.add(ekey)
                out.append({
                    "subject": s.value, "relation": rel, "object": o.value,
                    "confidence": meta["confidence"], "confidence_lb": meta["confidence_lb"],
                    "support": meta["support"], "contested": meta["contested"],
                    "years": (meta["first_year"], meta["last_year"]),
                    "hint_match": rel in hints,
                })
                for nxt in (s, o):
                    if nxt.value not in seen_nodes:
                        seen_nodes.add(nxt.value)
                        frontier.append((nxt, depth + 1))
        out.sort(key=lambda e: (e["hint_match"], e["confidence_lb"], e["years"][1] or 0),
                 reverse=True)
        return out

    def to_context(self, edges, limit=25):
        if not edges:
            return ""
        lines = ["Structured evidence from the literature graph:"]
        for e in edges[:limit]:
            subj = e["subject"].split("/")[-1]
            obj = e["object"].split("/")[-1]
            yr = e["years"][1]
            flag = " ⚠CONTESTED" if e["contested"] else ""
            lines.append(
                f"  - {subj} --[{e['relation']}]--> {obj} "
                f"(confidence {e['confidence']:.2f}, {e['support']} paper(s)"
                f"{f', latest {yr}' if yr else ''}){flag}"
            )
        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_store_query.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/kg/rdfstar_store.py tests/test_kg_store_query.py
git commit -m "feat(kg): weighted BFS query + prompt context serialization"
```

---

### Task 7: Serialization round-trip, merge, and factory

**Files:**
- Modify: `src/agent_graph/kg/rdfstar_store.py` (add `to_dict`, `from_dict`, `iter_claims`, `merge_from`)
- Create: `src/agent_graph/kg/__init__.py` content — add `get_knowledge_graph()` factory and `merge_graphs()` reducer
- Test: `tests/test_kg_store_persist.py`

**Interfaces:**
- Produces:
  - `OxigraphKG.to_dict() -> {"nquads": str}`; classmethod `OxigraphKG.from_dict(data) -> OxigraphKG`
  - `OxigraphKG.iter_claims()` → yields `(s_node, p_node, o_node, evidence_list)`
  - `OxigraphKG.merge_from(other)` → folds another KG's claims+evidence in, deduped by base triple + paper
  - `agent_graph.kg.get_knowledge_graph() -> KnowledgeGraph` (constructs `OxigraphKG`)
  - `agent_graph.kg.merge_graphs(a, b) -> KnowledgeGraph` (LangGraph reducer)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_store_persist.py
"""Round-trip serialization and branch merge."""
import pytest


def _ev(pmid, relevance=80, polarity="supports", year=2020):
    return {"paper_uri": f"pmid:{pmid}", "pmid": pmid, "paper_id": pmid,
            "pub_year": year, "relevance": relevance, "polarity": polarity, "snippet": "t"}


def test_to_dict_from_dict_roundtrip():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    kg = OxigraphKG()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    blob = kg.to_dict()
    kg2 = OxigraphKG.from_dict(blob)
    e1 = kg.query(["Imatinib"]); e2 = kg2.query(["Imatinib"])
    assert len(e1) == len(e2) == 1
    assert abs(e1[0]["confidence"] - e2[0]["confidence"]) < 1e-9


def test_merge_combines_evidence_without_double_counting():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    from agent_graph.kg import merge_graphs
    a = OxigraphKG(); a.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))
    b = OxigraphKG(); b.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("2"))
    b.add_relation("Imatinib", "drug", "treats", "CML", "disease", _ev("1"))  # overlap
    merged = merge_graphs(a, b)
    meta = merged._claim_meta("drug", "Imatinib", "treats", "disease", "CML")
    assert meta["support"] == 2  # papers 1 and 2, not 3


def test_merge_graphs_handles_none():
    from agent_graph.kg.rdfstar_store import OxigraphKG
    from agent_graph.kg import merge_graphs
    a = OxigraphKG(); a.add_relation("X", "drug", "treats", "Y", "disease", _ev("1"))
    assert merge_graphs(None, a) is a
    assert merge_graphs(a, None) is a


def test_factory_returns_protocol():
    from agent_graph.kg import get_knowledge_graph
    from agent_graph.kg.protocol import KnowledgeGraph
    kg = get_knowledge_graph()
    assert isinstance(kg, KnowledgeGraph)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_store_persist.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge_graphs'` / missing `to_dict`

- [ ] **Step 3: Add persistence + merge to the store**

Append to `class OxigraphKG` in `src/agent_graph/kg/rdfstar_store.py`:

```python
    def to_dict(self) -> dict:
        buf = io.BytesIO()
        self.store.dump(buf, ox.RdfFormat.N_QUADS)
        return {"nquads": buf.getvalue().decode("utf-8")}

    @classmethod
    def from_dict(cls, data: dict):
        store = ox.Store()
        store.load(io.BytesIO(data["nquads"].encode("utf-8")), format=ox.RdfFormat.N_QUADS)
        return cls(store)

    def iter_claims(self):
        for q in self.store.quads_for_pattern(None, _REIFIES, None, None):
            claim = q.object  # ox.Triple
            yield claim.subject, claim.predicate, claim.object, self._evidence(q.subject)

    def merge_from(self, other):
        for s, p, o, ev in other.iter_claims():
            claim = ox.Triple(s, p, o)
            self.store.add(ox.Quad(s, p, o))
            r = self._reifier_for(claim)
            if r is None:
                r = ox.BlankNode()
                self.store.add(ox.Quad(r, _REIFIES, claim))
                existing = []
            else:
                existing = self._evidence(r)
            self._write_annotations(r, existing + ev)
            rel = p.value.replace(ont.KG, "")
            self._flag_conflicts(rel, s, o)
        return self
```

- [ ] **Step 4: Write the factory + reducer in the package init**

Replace the empty `src/agent_graph/kg/__init__.py` with:

```python
# src/agent_graph/kg/__init__.py
"""Domain knowledge graph package."""
from agent_graph.kg.protocol import KnowledgeGraph
from agent_graph.kg.rdfstar_store import OxigraphKG


def get_knowledge_graph() -> KnowledgeGraph:
    """Factory — the single place that picks the backend."""
    return OxigraphKG()


def merge_graphs(a, b):
    """LangGraph reducer: fold parallel map-reduce branches into one graph."""
    if a is None:
        return b if b is not None else OxigraphKG()
    if b is None:
        return a
    a.merge_from(b)
    return a
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_store_persist.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/agent_graph/kg/rdfstar_store.py src/agent_graph/kg/__init__.py tests/test_kg_store_persist.py
git commit -m "feat(kg): N-Quads round-trip, branch merge reducer, backend factory"
```

---

### Task 8: graph_builder_node

**Files:**
- Modify: `src/agent_graph/nodes.py` (add `graph_builder_node` + helpers near the other nodes; add imports at top)
- Test: `tests/test_kg_builder_node.py`

**Interfaces:**
- Consumes: `agent_graph.kg.get_knowledge_graph`, `agent_graph.kg.extract.TripletExtraction/ScientificTriplet`, `agent_graph.llm.get_llm`.
- Produces: `graph_builder_node(state) -> {"knowledge_graph": KnowledgeGraph}`. Reads `state["papers"]` and optional `state.get("knowledge_graph")`. For each paper: extracts triplets (best-effort), adds them with evidence, plus free `author --wrote--> paper` edges. LLM is obtained via module-level `get_llm` (monkeypatchable in tests).
- Helper produced: `_pub_year(published: str) -> int | None` (parses a leading 4-digit year).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_builder_node.py
"""Tests for graph_builder_node with a stubbed LLM."""
import pytest


class _FakeStructured:
    def __init__(self, result): self._result = result
    def invoke(self, _messages): return self._result


class _FakeLLM:
    def __init__(self, result=None, raises=False):
        self._result, self._raises = result, raises
    def with_structured_output(self, _schema):
        if self._raises:
            class _Boom:
                def invoke(self, _m): raise RuntimeError("extraction failed")
            return _Boom()
        return _FakeStructured(self._result)


def _paper():
    return {"id": "P1", "pmid": "111", "title": "Imatinib in CML",
            "authors": ["Alice Smith", "Bob Jones"], "published": "2020-05-01",
            "summary": "Imatinib treats CML.", "url": "u", "relevance_score": 90}


def test_builder_adds_claims_and_wrote_edges(monkeypatch):
    import agent_graph.nodes as nodes
    from agent_graph.kg.extract import TripletExtraction, ScientificTriplet
    result = TripletExtraction(triplets=[
        ScientificTriplet(subject="Imatinib", subject_type="drug",
                          relation="treats", object="CML", object_type="disease")
    ])
    monkeypatch.setattr(nodes, "get_llm", lambda **kw: _FakeLLM(result=result))
    out = nodes.graph_builder_node({"papers": [_paper()]})
    kg = out["knowledge_graph"]
    assert kg._claim_meta("drug", "Imatinib", "treats", "disease", "CML")["support"] == 1
    # free authorship edge: Alice wrote pmid:111
    edges = kg.query(["Alice Smith"])
    assert any(e["relation"] == "wrote" for e in edges)


def test_builder_skips_papers_on_extraction_failure(monkeypatch):
    import agent_graph.nodes as nodes
    monkeypatch.setattr(nodes, "get_llm", lambda **kw: _FakeLLM(raises=True))
    out = nodes.graph_builder_node({"papers": [_paper()]})
    # pipeline not blocked; authorship edges still added from metadata
    kg = out["knowledge_graph"]
    assert kg.query(["Alice Smith"])  # wrote edge present despite extraction failure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_builder_node.py -v`
Expected: FAIL — `AttributeError: module 'agent_graph.nodes' has no attribute 'graph_builder_node'`

- [ ] **Step 3: Add imports + node to nodes.py**

Add near the top imports of `src/agent_graph/nodes.py`:

```python
from agent_graph.kg import get_knowledge_graph
from agent_graph.kg.extract import TripletExtraction
from agent_graph.kg.ontology import paper_uri
```

Add this node function (place it after `pubmed_researcher_node`, before `summarizer_node`):

```python
def _pub_year(published: str):
    """Parse a leading 4-digit year from a published-date string."""
    import re
    if not published:
        return None
    m = re.search(r"(19|20)\d{2}", str(published))
    return int(m.group(0)) if m else None


def graph_builder_node(state):
    """Extract ontology-constrained claims from papers into the knowledge graph.

    Best-effort: a paper whose extraction fails is skipped, never blocking the
    pipeline. Also materializes free author --wrote--> paper edges from metadata.
    Runs after the researcher, before the summarizer.
    """
    papers = state.get("papers", [])
    kg = state.get("knowledge_graph") or get_knowledge_graph()
    llm = get_llm(temperature=0).with_structured_output(TripletExtraction)

    for p in papers:
        p_uri = paper_uri(paper_id=p.get("id"), pmid=p.get("pmid"))
        relevance = p.get("relevance_score", 50)
        year = _pub_year(p.get("published"))

        # claim extraction (best-effort)
        try:
            extraction = llm.invoke([
                SystemMessage(content=EXTRACTION_PROMPT_HEADER),
                HumanMessage(content=f"{p.get('title','')}\n\n{p.get('summary','')}"),
            ])
            triplets = extraction.triplets
        except Exception as exc:
            logging.warning(f"⚠️  KG extraction failed for paper {p.get('id')}: {exc}")
            triplets = []

        for t in triplets:
            kg.add_relation(
                t.subject, t.subject_type, t.relation, t.object, t.object_type,
                {"paper_uri": p_uri, "pmid": p.get("pmid"), "paper_id": p.get("id"),
                 "pub_year": year, "relevance": relevance, "polarity": t.polarity,
                 "snippet": p.get("title", "")[:160]},
            )

        # free authorship edges from existing metadata
        for author in p.get("authors", [])[:5]:
            kg.add_relation(
                author, "author", "wrote", p_uri, "paper",
                {"paper_uri": p_uri, "pmid": p.get("pmid"), "paper_id": p.get("id"),
                 "pub_year": year, "relevance": relevance, "polarity": "supports",
                 "snippet": "authorship"},
            )

    return {"knowledge_graph": kg}
```

Add the prompt header import alias near the other module constants (the extraction prompt lives in `extract.py`; reference it directly):

```python
from agent_graph.kg.extract import EXTRACTION_PROMPT as EXTRACTION_PROMPT_HEADER
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_builder_node.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/nodes.py tests/test_kg_builder_node.py
git commit -m "feat(kg): graph_builder_node — best-effort claim + authorship extraction"
```

---

### Task 9: Wire into the demo graph + inject context into the summarizer

**Files:**
- Modify: `src/agent_graph/state.py` (add `knowledge_graph` field + reducer)
- Modify: `src/agent_graph/nodes.py` (`summarizer_node` — inject graph context)
- Modify: `src/agent_graph/graph.py` (`create_demo_graph` — splice `graph_builder`)
- Test: `tests/test_kg_wiring.py`

**Interfaces:**
- Consumes: `graph_builder_node` (Task 8), `merge_graphs`/`KnowledgeGraph` (Task 7).
- Produces: `InternalState["knowledge_graph"]` reducer-merged field; `create_demo_graph()` with node order `clarifier → pubmed_researcher → graph_builder → summarizer → dual_audience → hitl_approval`; `summarizer_node` prompt includes `to_context()` when a graph is present.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kg_wiring.py
"""Wiring tests: state field + summarizer context injection + demo graph shape."""
import pytest


def test_state_has_knowledge_graph_field():
    from agent_graph.state import InternalState
    assert "knowledge_graph" in InternalState.__annotations__


def test_demo_graph_includes_graph_builder():
    from agent_graph.graph import create_demo_graph
    g = create_demo_graph()
    assert "graph_builder" in g.get_graph().nodes


def test_summarizer_injects_graph_context(monkeypatch):
    import agent_graph.nodes as nodes
    from agent_graph.kg import get_knowledge_graph

    captured = {}

    class _LLM:
        def invoke(self, messages):
            captured["messages"] = messages
            from langchain_core.messages import AIMessage
            return AIMessage(content="## Summary\n• ok [Paper 1]")

    monkeypatch.setattr(nodes, "get_llm", lambda **kw: _LLM())

    kg = get_knowledge_graph()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease",
                    {"paper_uri": "pmid:1", "pmid": "1", "paper_id": "1",
                     "pub_year": 2020, "relevance": 90, "polarity": "supports", "snippet": "t"})
    state = {
        "papers": [{"id": "1", "pmid": "1", "title": "T", "authors": ["A"],
                    "published": "2020", "summary": "s", "url": "u", "relevance_score": 90}] * 3,
        "query": "imatinib cml", "iteration": 0, "max_iterations": 2,
        "knowledge_graph": kg,
    }
    nodes.summarizer_node(state)
    blob = "\n".join(str(getattr(m, "content", "")) for m in captured["messages"])
    assert "literature graph" in blob and "imatinib" in blob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kg_wiring.py -v`
Expected: FAIL — `knowledge_graph` not in annotations / `graph_builder` not in nodes.

- [ ] **Step 3: Add the state field + reducer**

In `src/agent_graph/state.py`, add the import after the existing `get_llm` import:

```python
from agent_graph.kg import merge_graphs
```

Add the field to `InternalState` (after the `approved` line):

```python
    knowledge_graph: Annotated[NotRequired[object], merge_graphs]  # domain KG (OxigraphKG)
```

- [ ] **Step 4: Inject context in summarizer_node**

In `src/agent_graph/nodes.py`, inside `summarizer_node`, immediately after `papers_context = "\n\n".join([...])` (the block ending around line 387), insert:

```python
    graph_context = ""
    kg = state.get("knowledge_graph")
    if kg is not None:
        edges = kg.query([query], min_confidence=0.4, max_depth=2)
        graph_context = kg.to_context(edges)
```

Then change the summarizer `HumanMessage` to append the graph context. Replace:

```python
        HumanMessage(content=f"""Original question: {query}
        
Papers found:
{papers_context}

Generate a structured summary.""", name="User")
```

with:

```python
        HumanMessage(content=f"""Original question: {query}
        
Papers found:
{papers_context}

{graph_context}

Generate a structured summary.""", name="User")
```

- [ ] **Step 5: Splice graph_builder into create_demo_graph**

In `src/agent_graph/graph.py`, add `graph_builder_node` to the import list from `agent_graph.nodes` (append to the existing parenthesized import):

```python
    graph_builder_node,
```

In `create_demo_graph`, add the node and rewire the two edges. Replace:

```python
    workflow.add_node("pubmed_researcher", pubmed_researcher_node)
    workflow.add_node("summarizer", summarizer_node)
```

with:

```python
    workflow.add_node("pubmed_researcher", pubmed_researcher_node)
    workflow.add_node("graph_builder", graph_builder_node)
    workflow.add_node("summarizer", summarizer_node)
```

and replace:

```python
    workflow.add_edge("pubmed_researcher", "summarizer")
```

with:

```python
    workflow.add_edge("pubmed_researcher", "graph_builder")
    workflow.add_edge("graph_builder", "summarizer")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_kg_wiring.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Run the full KG test suite + existing suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all KG tests pass; pre-existing tests (`test_llm_factory`, `test_pubmed`, `test_schemas`) remain green.

- [ ] **Step 8: Commit**

```bash
git add src/agent_graph/state.py src/agent_graph/nodes.py src/agent_graph/graph.py tests/test_kg_wiring.py
git commit -m "feat(kg): wire graph_builder into demo graph + inject context into summarizer"
```

---

## Self-Review

**Spec coverage:**
- §3 module layout → Tasks 2–7 create every file (`ontology, confidence, extract, protocol, rdfstar_store, __init__`).
- §4 pyoxigraph + RDF 1.2 reifier → Task 1 (dep) + Task 5 (`rdf:reifies` reifier model).
- §5 frozen ontology, FUNCTIONAL={}, MUTEX → Task 2 + Task 5 `_flag_conflicts`/`_functional_violation`.
- §6 URI scheme, no synonym merging, extraction schema → Task 2 (`entity_uri`/`paper_uri`), Task 4.
- §7 Protocol methods, weighted BFS, as_of_year, to_dict/from_dict → Tasks 5–7.
- §8 Beta-Bernoulli + confidence_lb → Task 3, consumed in Task 5.
- §9 reserved work → untouched by design; mechanism seams (`FUNCTIONAL`, `as_of_year`) present.
- §10 node wiring (state field+reducer, graph_builder, summarizer injection, graph.py) → Tasks 7–9.
- §11 testing → every task is TDD; Task 9 Step 7 runs full suite incl. pre-existing.
- §12 dependency → Task 1.

**Placeholder scan:** No TBD/TODO; every code step shows complete code.

**Type consistency:** The evidence boundary type `Evidence` (Task 4) has fields `paper_uri, pmid, paper_id, pub_year, relevance, polarity, snippet`; `add_relation` (Task 5) parses any dict into it via `model_validate(...).model_dump()`, so the internal dict shape consumed by `_dedupe`/`_write_annotations`/`beta_params` is exactly `Evidence.model_dump()`. Call sites in Tasks 5/7/8/9 build dicts with those keys (relevance always provided). `_claim_meta`/`_reifier_meta` dict keys (`confidence, confidence_lb, support, contested, first_year, last_year`) are produced in Task 5 and consumed in Tasks 6–9. `query()` edge-dict keys (`subject, relation, object, confidence, confidence_lb, support, contested, years, hint_match`) defined in Task 6, consumed by `to_context` (Task 6) and the summarizer (Task 9). `merge_graphs`/`get_knowledge_graph` defined in Task 7, imported in Tasks 8–9.

**Parse-don't-validate coverage:** untrusted inputs are parsed into trusted types at exactly two boundaries — LLM output → `TripletExtraction` (existing pattern, Task 8) and evidence blob → `Evidence` (Task 5). No raw dict flows past `add_relation`. The `VerifiedClaim`-as-distinct-type idea is intentionally NOT here — it belongs to the reserved faithfulness/reconciliation layer (spec §9), not v1.

**Scope:** One package, one pipeline integration — single coherent plan.
