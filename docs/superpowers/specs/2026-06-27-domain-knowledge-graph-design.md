# Domain Knowledge Graph for Scientific Literature — Design

**Date:** 2026-06-27
**Status:** Approved design, pre-implementation
**Component:** `src/agent_graph/kg/` (new package)

## 1. Motivation & Scope

Scientific Graph Agent (SGA) currently represents retrieved papers as flat dicts,
reranked by a hybrid SPECTER + BGE scorer. There is no structured representation of
the *claims* those papers make, so cross-paper evidence ("how many studies, how
strongly, do they agree?") cannot be reasoned over or surfaced.

This design adds an **additive** domain knowledge graph that extracts
`(subject)-[relation]->(object)` claims from paper abstracts, accumulates
per-claim evidence across papers, and injects the resulting weighted, structured
context into the summarizer/dual-audience nodes. It does **not** refactor the
flat-paper pipeline (`reranker.py`, `keep_top_papers`) — nodes call the graph
optionally.

### Domain framing (drives every decision below)

A literature graph's "conflicts" are **signal, not error**. Two papers disagreeing
about a claim is the scientific content a research agent exists to surface — the
opposite of a clinical record, where a contradiction is a defect to resolve.
Consequently this design *retains and weights* disagreement rather than resolving
it, and demotes the clinical-grade reconciliation/bitemporal machinery to reserved
future work (Section 9).

### Forward-looking constraint

This graph is intended to later back **clinical pathology report** structuring and
**LOINC/SNOMED** terminology reconciliation. Those terminologies are published as
RDF/OWL, and clinical data genuinely needs bitemporality, contradiction resolution,
and entity reconciliation. The architecture is therefore chosen so that machinery
is an *additive bolt-on*, never a rewrite: see the RDF* decision (Section 4) and
the reserved-work seams (Section 9).

## 2. Goals / Non-Goals

**Goals (v1)**
- Extract ontology-constrained claim triples from paper abstracts.
- Accumulate cross-paper evidence per claim with a principled confidence model.
- Flag (not resolve) logically contradictory claims.
- Inject weighted, contested-aware structured context into generation nodes.
- Keep the storage backend swappable behind an interface.

**Non-Goals (v1) — reserved, see Section 9**
- Bitemporality / valid-time (one publication-date axis only).
- Contradiction *resolution* or human-review routing.
- LLM reconciliation pass; entity/synonym resolution (`imatinib` ≠ `gleevec`).
- `cites` edges (needs CrossRef/Semantic Scholar).
- Persistent/durable store; SPARQL-star analytical queries.
- Graph-centrality reranking.

## 3. Module Layout

```
src/agent_graph/kg/
├── __init__.py
├── protocol.py      # KnowledgeGraph Protocol — the ONLY thing nodes import
├── ontology.py      # frozen enums: EntityType, RelationType, FUNCTIONAL, MUTEX; URI minting
├── confidence.py    # Beta-Bernoulli update + readout (pure functions)
├── rdfstar_store.py # rdflib-star implementation of the Protocol
└── extract.py       # ScientificTriplet / TripletExtraction Pydantic + extraction prompt
```

Nodes import only `protocol.py` and `extract.py`. Swapping the backend (rdflib →
oxigraph) touches only `rdfstar_store.py` plus a factory.

## 4. Storage Decision: RDF* behind an interface

**Decision:** RDF* (RDF-star) semantics, abstracted behind a `KnowledgeGraph`
Protocol. v1 implementation is in-memory **rdflib** with quoted-triple support;
**oxigraph** (durable, Rust-backed, real SPARQL-star) is a reserved swap-in.

**Why RDF\*** over a property-graph (NetworkX) or plain RDF:
- Per-claim metadata (confidence, evidence, provenance, year range) is a
  *statement about a statement* — which RDF* models natively
  (`<<s p o>> :confidence 0.82`) and plain RDF can only express via verbose
  blank-node reification (the "messy versioned ontology" failure mode we are
  avoiding).
- The clinical endgame (Section 1) makes standards interop (LOINC/SNOMED/FHIR as
  RDF/OWL, SHACL validation, SPARQL-star) a real future asset rather than overhead.

**Why behind an interface:** rdf-star tooling in Python is the immature part. The
Protocol lets the rdflib-vs-oxigraph choice be revisited without touching node
wiring.

**Trade-off accepted:** rdflib's SPARQL-star is slow/immature, so graph *traversal*
(Section 7) is implemented in Python over the triples; SPARQL-star is reserved for
later analytical queries.

## 5. Ontology (frozen for v1)

Declared once in `ontology.py`; the extractor is constrained to these via the
Pydantic schema (Section 6), so the LLM physically cannot mint off-ontology URIs.

**Entity types (8):** `drug, disease, gene, biomarker, method, population, paper, author`

**Predicates:**
- *Claim (9):* `treats, causes, associated_with, inhibits, increases_risk_of,
  decreases_risk_of, measured_by, subtype_of, studied_in`
- *Bibliographic:* `wrote` (author→paper; materialized free from existing paper
  dicts). **Reserved:** `cites` (paper→paper) needs CrossRef/Semantic Scholar
  reference data SGA does not fetch; `affiliated_with` (author→org) needs a 9th
  `organization` entity type and reliable affiliation metadata (inconsistent in
  ArXiv/PubMed) — both deferred to keep the v1 entity set at 8 and avoid unreliable
  extraction.
- *Provenance:* `asserted_by` (claim→paper) — the RDF* edge-on-edge link that makes
  confidence/evidence auditable.
- *Contradiction:* `contradicts` (claim→claim) — explicit, complementing auto MUTEX
  flagging.

**Constraints:**
- `FUNCTIONAL = {}` — nothing is single-valued in a literature graph. A literature
  graph aggregates many papers' claims, correctly yielding multiple objects per
  (subject, relation); even `subtype_of` allows multiple parents. A FUNCTIONAL
  declaration would raise false `functional-violation` flags on normal aggregation,
  polluting the contradiction channel. The detection **mechanism is retained** so a
  genuinely single-valued predicate (likely in the clinical extension) is a one-line
  re-add.
- `MUTEX = {(increases_risk_of, decreases_risk_of)}` — logically opposed predicates
  on the same pair are genuine disagreement worth flagging even in literature.
  Flag-not-reject: both edges persist and are marked `contested`.

## 6. Entity Identity & Extraction

**URI scheme (v1 entity resolution = pure string normalization):**
- Entities: `kg:{type}/{slug}`, slug = lowercased, whitespace/punctuation-normalized
  name (e.g. `kg:drug/imatinib`). **No synonym merging** — `imatinib` and `gleevec`
  are distinct nodes. This is the largest v1 limitation; synonym/entity resolution is
  reserved (Section 9) and is the natural insertion point for LOINC/SNOMED mapping.
- Papers: reuse real identifiers — `pmid:12345` / `arxiv:2401.xxxxx` (free,
  globally meaningful provenance).
- Claims: quoted triples `<< kg:drug/imatinib kg:treats kg:disease/cml >>`,
  annotated with `confidence, confidence_lb, alpha, beta, support, refute,
  first_year, last_year, contested`, plus one `asserted_by` link per supporting
  paper. Per-evidence detail (relevance, polarity, snippet) hangs off the
  `asserted_by` link.

**Extraction schema** (`extract.py`, enforced at the structured-output boundary):

```python
class ScientificTriplet(BaseModel):
    subject: str
    subject_type: Literal[<8 entity types>]
    relation: Literal[<claim + bibliographic predicates>]
    object: str
    object_type: Literal[<8 entity types>]
    polarity: Literal["supports", "refutes"] = "supports"

class TripletExtraction(BaseModel):
    triplets: list[ScientificTriplet] = Field(default_factory=list)
```

The prompt instructs extraction of on-ontology relations from the abstract only,
with `polarity="refutes"` when the paper reports a null/negative finding for a
claim.

## 7. KnowledgeGraph Protocol

```python
class KnowledgeGraph(Protocol):
    def add_relation(self, subject, subject_type, relation, obj, object_type,
                     evidence: dict) -> str | None: ...   # returns conflict note or None
    def query(self, entities, relation_hints=None, max_depth=2,
              min_confidence=0.0, as_of_year=None) -> list[dict]: ...
    def to_context(self, edges, limit=25) -> str: ...
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph": ...
```

- `add_relation` upserts the claim, appends evidence, recomputes confidence
  (Section 8), and runs `_flag_conflicts` (MUTEX + retained-but-empty FUNCTIONAL).
- `query` does a **weighted BFS** in Python over the triples: from seed entities,
  expand up to `max_depth` hops in both directions, filtering by `min_confidence`
  and `as_of_year` (the single publication-date axis), ranking results by
  `confidence_lb` then hint-match then recency. Returns plain dicts.
- `to_context` serializes ranked edges into a prompt block, marking `⚠CONTESTED`.
- `to_dict`/`from_dict` serialize the graph into the LangGraph sqlite checkpoint.

## 8. Confidence: Beta-Bernoulli

Each claim is a Beta-Bernoulli belief. Evidence contributes relevance-weighted
fractional pseudo-counts:

```
w(e)  = 0.3 + 0.65 * (relevance / 100)        # per-paper weight in ~[0.3, 0.95]
alpha = PRIOR_A + Σ w(e) over polarity=supports
beta  = PRIOR_B + Σ w(e) over polarity=refutes
confidence    = alpha / (alpha + beta)                       # posterior mean
var           = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
confidence_lb = max(0, confidence - 1.64 * sqrt(var))        # ranking signal
```

Prior `Beta(1, 1)` (uniform; raise `PRIOR_B` to be more skeptical). Ranking uses
`confidence_lb` so thin evidence is penalized and refuting papers pull confidence
down. Pure functions in `confidence.py`.

## 9. Reserved Future Work (explicit seams)

| Reserved | Seam that keeps it additive |
|---|---|
| Second time axis / full bitemporality | Evidence dict + `_recompute`; query already has `as_of_year` |
| Contradiction resolution / human-review routing | `_flag_conflicts` returns a note today; route it later |
| LLM reconciliation pass | A new node downstream of `graph_builder`; nothing depends on its absence |
| Entity/synonym resolution + LOINC/SNOMED mapping | URI minting in `ontology.py`; the single normalization seam |
| `cites` edges | Reserved predicate; add CrossRef/S2 fetch + edges |
| `affiliated_with` + `organization` entity type | Reserved predicate; add 9th entity type + affiliation source |
| Durable store (oxigraph) | Protocol + factory; swap `rdfstar_store.py` |
| SPARQL-star analytical queries | rdflib-star already stores quoted triples |
| Graph-centrality reranking | `query` returns structured edges; feed into `reranker.py` |

## 10. Node Wiring

- **`state.py`:** add `knowledge_graph: Annotated[NotRequired[KnowledgeGraph],
  merge_graphs]`. `merge_graphs` reducer folds parallel map-reduce branches by
  replaying evidence.
- **`graph_builder_node` (new):** runs after researcher, before summarizer. Per
  paper: `get_llm(temperature=0).with_structured_output(TripletExtraction)`
  constrained to the ontology → `add_relation` with evidence
  `{paper_id, pmid, pub_year, relevance, polarity, snippet}`; also materializes free
  `author --wrote--> paper` edges from the paper dict. **Best-effort**: extraction
  failure on a paper is caught and skipped, never blocking the pipeline. Idempotent
  per `paper_id` (evidence append + monotonic recompute).
- **`summarizer_node` / `dual_audience_node`:** pull a weighted subgraph via
  `query()` for the query entities (`min_confidence ≈ 0.4`), inject `to_context()`
  (with `⚠CONTESTED` flags) into the system prompt alongside the existing paper
  list.
- **`graph.py`:** splice `researcher → graph_builder → summarizer` in
  `create_demo_graph` (and the other builders). The existing `should_continue`
  retry edge to the researcher is unchanged; `graph_builder` re-runs on new papers
  and accumulates evidence.

## 11. Testing

- `confidence.py`: pure unit tests (monotonicity, refutation lowers mean,
  thin-evidence penalty via `confidence_lb`).
- `ontology.py`: URI minting/normalization, MUTEX detection, empty-FUNCTIONAL no-op.
- `rdfstar_store.py`: `to_dict`/`from_dict` round-trip; BFS traversal + ranking on a
  hand-built graph; `as_of_year` and `min_confidence` filtering.
- `graph_builder_node`: stubbed LLM returning fixed `TripletExtraction`; asserts
  edges + free `wrote` edges added, extraction-failure path skips gracefully.
- No network in tests.

## 12. Dependencies

Add `rdflib` (pure-Python, supports RDF-star/quoted triples) to `pyproject.toml`.
No compiled or service dependencies in v1.
