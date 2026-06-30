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
├── rdfstar_store.py # pyoxigraph (RDF 1.2 reifier) implementation of the Protocol
└── extract.py       # ScientificTriplet / TripletExtraction Pydantic + extraction prompt
```

Nodes import only `protocol.py` and `extract.py`. Swapping the backend touches only
`rdfstar_store.py` plus a factory.

## 4. Storage Decision: RDF* behind an interface

**Decision:** RDF* (RDF 1.2 / RDF-star) semantics, abstracted behind a
`KnowledgeGraph` Protocol. v1 implementation is **pyoxigraph** (Rust-backed, real
RDF-star + SPARQL-star, durable on-disk store).

**Empirically verified during planning (2026-06-27), correcting an earlier
assumption:**
- **rdflib 7.6.0 has NO RDF-star** — no quoted-triple term type, no Turtle-star
  parser. The experimental 6.x support is gone. An "rdflib-star" v1 is not
  implementable.
- **The old RDF-star syntax (quoted-triple-as-subject, `<<s p o>> :conf 0.8`) is
  obsolete.** W3C **RDF 1.2** replaced it with **reification**: a reifier node
  links to a *triple term* via `rdf:reifies`, and annotations hang on the reifier.
- **pyoxigraph 0.5.9 implements the current model natively** (`RdfFormat.supports_rdf_star`
  is true), with SPARQL-star (`<<( s p o )>>` query syntax verified) and a durable
  Store. This is the real RDF* we want, via the standards-current API.

**Why RDF\*** over a property-graph (NetworkX) or plain manual reification:
- Per-claim metadata (confidence, evidence, provenance, year range, and the
  reserved bitemporal valid/transaction times) is a *statement about a statement* —
  modeled natively by the reifier node.
- The clinical endgame (Section 1) makes standards interop a real future asset:
  `loinc.ttl` and SNOMED-OWL can load into the *same* oxigraph store as the claim
  graph, so SPARQL can traverse a LOINC/SNOMED hierarchy AND attach
  `term --mapped_to--> code` edges with RDF-star provenance in one engine — which a
  property graph cannot do without re-implementing terminology import. (Note: RDF*
  is the unifying *substrate*; the reconciliation matching logic + any FHIR
  terminology server remain separate work — see Section 9.)

**Why behind an interface:** RDF-star tooling churn (the rdflib/RDF-1.2 surprise
above) is exactly the risk the Protocol absorbs. Swapping stores later touches only
`rdfstar_store.py` + a factory; node wiring is unaffected.

**Representation (RDF 1.2 reifier model):** each claim is a base triple
`(s, p, o)` plus a reifier (blank node) with `(reifier, rdf:reifies, <<s p o>>)`
and annotation triples `(reifier, kg:confidence, …)`, `(reifier, kg:asserted_by,
pmid)`, etc. The reserved bitemporal axis (Section 9) is additive: it is just more
annotation triples on the same reifier.

**Trade-off accepted:** pyoxigraph is a compiled dependency. Graph *traversal*
(Section 7) uses oxigraph's native quad iteration / SPARQL; SPARQL-star analytical
queries are available now but only a thin slice is used in v1.

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
- Claims: base triple `(kg:drug/imatinib, kg:treats, kg:disease/cml)` plus a reifier
  blank node `r` with `(r, rdf:reifies, <<s p o>>)` (RDF 1.2). Aggregate annotations
  hang on `r`: `kg:confidence, kg:confidence_lb, kg:alpha, kg:beta, kg:support,
  kg:refute, kg:first_year, kg:last_year, kg:contested`. Each supporting paper adds
  one `(r, kg:asserted_by, <paper_uri>)`. Per-evidence detail (relevance, polarity,
  snippet) is modeled as its own reifier on the `asserted_by` statement, or — for v1
  simplicity — stored compactly as a JSON literal on `kg:evidence` (decided in the
  plan; both are pure additions of annotation triples).

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
- `query` does a **weighted BFS** over the oxigraph store: from seed entities,
  expand up to `max_depth` hops in both directions via `store.quads_for_pattern`,
  joining each base triple to its reifier annotations, filtering by `min_confidence`
  and `as_of_year` (the single publication-date axis), ranking results by
  `confidence_lb` then hint-match then recency. Returns plain dicts.
- `to_context` serializes ranked edges into a prompt block, marking `⚠CONTESTED`.
- `to_dict`/`from_dict` serialize the store (N-Quads via `store.dump`/`Store.load`)
  so the graph rides in the LangGraph sqlite checkpoint as a string.

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

Two distinct layers are reserved — keep them straight (they do not overlap):
*knowledge-data-model* items (bitemporality, reconciliation) live on the graph;
*execution-audit* items (FHIR AuditEvent) live at the LangGraph checkpoint layer and
are backend-independent. Bitemporality is the one item that depends on the RDF 1.2
reifier we build in v1; AuditEvent does not.

| Reserved | Layer | Seam that keeps it additive |
|---|---|---|
| Second time axis / full bitemporality (valid-time + transaction-time) | data-model | `valid_from/valid_to/tx_from/tx_to` are more annotation triples on the existing reifier; `query` already has `as_of_year` |
| Contradiction resolution / human-review routing | data-model | `_flag_conflicts` returns a note today; route it later |
| LLM reconciliation pass | data-model | A new node downstream of `graph_builder`; nothing depends on its absence |
| Entity/synonym resolution + LOINC/SNOMED/RxNorm mapping | data-model | URI minting in `ontology.py` is the single normalization seam; load `loinc.ttl`/SNOMED-OWL/RxNorm into the same store. (Matching logic + optional FHIR terminology server are separate work, not free from RDF*.) |
| `VerifiedClaim` (claim faithfulness vs. source) | data-model | `verified: float` field on `Evidence`, multiplied into `evidence_weight` in `confidence.py`; see §9.1 |
| Correlation discount (independence leak) | data-model | Distinct from `VerifiedClaim`; needs `cites`/co-authorship data + a novelty/hierarchical model; see §9.1 |
| `cites` edges | data-model | Reserved predicate; add CrossRef/S2 fetch + edges |
| `affiliated_with` + `organization` entity type | data-model | Reserved predicate; add 9th entity type + affiliation source |
| FHIR AuditEvent (who/what/when did an action) | execution-audit | Overlaps LangGraph checkpointing, not bitemporal edges; emit from a checkpoint listener — independent of the graph backend |
| JSONB persistence of approved outputs | execution-audit | Store HITL-approved summaries (and their `validation_errors`) as JSONB in Postgres for audit/query; the after-approval storage layer that pairs with FHIR AuditEvent. The *pre*-HITL JSON Schema gate is already implemented (see EXTENSION_NOTES §7) — distinct concern: validation vs. storage |
| JSON-LD output mapping | data-model | Give the agent's approved summary an `@context` mapping its entities to RDF + LOINC/SNOMED/RxNorm URIs, so an approved summary becomes a node in this KG; the semantic-interop bridge between agent JSON and the claim graph. Depends on the reconciliation seam (§9.1) for the target URIs |
| SPARQL-star analytical queries | data-model | oxigraph already stores RDF-star; widen query use |
| Graph-centrality reranking | data-model | `query` returns structured edges; feed into `reranker.py` |

### 9.1 Clinical extension: separate plans, ordered, with two non-overlapping confidence leaks

The clinical workstream (`VerifiedClaim` + terminology reconciliation) is **two separate
specs/plans**, not one, because the two are orthogonal (claim-faithfulness vs.
entity-identity) and have different seams. Both depend on the v1 KG. **Order:
reconciliation first, then `VerifiedClaim`** — entity identity is upstream of claim
trust: if `imatinib` and `gleevec` are separate nodes, cross-source Beta-Bernoulli
aggregation fragments and any faithfulness guarantee over them means little.
(imatinib↔gleevec is specifically an **RxNorm** brand/generic relation, not LOINC/SNOMED;
prefer a confidence-weighted `:maps_to` soft link to the canonical *ingredient* concept
over a destructive URI merge, so provenance survives.)

**Two independent leaks in "many papers agree → high confidence" — do not conflate:**

- **Leak A — faithfulness.** Is each vote real? Failure: the extractor hallucinates/
  misattributes a claim to a paper. A fake vote inflates α. **`VerifiedClaim` closes
  this** by checking each evidence→claim link against source, so a fake vote gets
  `verified→0` and never enters α.
- **Leak B — independence.** Are the votes distinct observations? Failure: N papers
  tracing to one primary study / same lab / reviews restating a finding — each vote is
  *faithful* but correlated, so counting them as N independent α-increments overstates
  confidence. **`VerifiedClaim` does NOT close this** — verification is per-edge;
  correlation is a relationship *between* edges. A fully verified graph can still be
  badly over-confident via a citation cascade. Closing Leak B needs different inputs
  (citation/co-authorship — `cites` is reserved) and different math (novelty discount,
  or a hierarchical/random-effects model grouping papers by source-cluster). It is its
  own reserved item, NOT a `VerifiedClaim` extension.

**Probability-flow limits.** The Beta mean `α/(α+β)` is structurally bounded in (0,1)
with the `Beta(1,1)` prior — it asymptotes, never saturates. Verification's role is to
bound confidence at the *grounded* evidence mass (Leak A), not to change that ceiling.

**`VerifiedClaim` spec contents (when authored):**
- *v1:* scalar `verified ∈ [0,1]` multiplier on `evidence_weight` (`verified * (0.3 +
  0.65*relevance/100)`); `VerifiedClaim` = an `Evidence`/`WeightedEdge` whose `verified`
  is populated by the verification pass.
- *Documented extension (same spec):* hierarchical **Beta-Binomial** — model
  verification as its own Beta belief (verified k of n checks) and propagate the
  uncertainty instead of collapsing to a scalar. Shares the same `verified` field; this
  is the "if it matters clinically" upgrade, kept in-spec to record why the scalar is a
  deliberate simplification. The correlation discount (Leak B) explicitly does **not**
  belong in this spec.

**Bitemporality is a third, independent axis — not part of either clinical plan.**
Reconciliation fixes entity identity; `VerifiedClaim` fixes claim faithfulness;
bitemporality versions facts over time (valid-time + transaction-time). Its trigger is
distinct: it activates only when EHR-style **temporal clinical records** enter the graph
(patient-level facts that change or arrive late), NOT when terminology mapping or
faithfulness is needed. For the literature agent the single publication-date axis
(`as_of_year`) suffices, so bitemporality may never activate in this codebase at all.
It neither blocks nor is blocked by the other two — ordering among the three is
independent. No plan until EHR-style temporal data is in scope; the reifier-annotation
seam is already documented above.

**None of the three plans is authored yet — deliberately.** A bite-sized TDD plan
requires exact file paths, signatures, and seams against *existing* code; writing these
against the unbuilt v1 KG would force placeholders (a writing-plans failure) and would
prematurely freeze design choices that still need their own brainstorming (resolver
mechanism + terminology source format for reconciliation; verification check method +
source-matching for `VerifiedClaim`). The correct record at this stage is this spec
sketch, not three speculative plans. When the clinical workstream is green-lit, the
sequence is: build v1 KG → brainstorm + spec → plan → implement reconciliation → repeat
for `VerifiedClaim`; bitemporality only if temporal records arrive.

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

Add `pyoxigraph>=0.5` to `pyproject.toml` — Rust-backed RDF store with native
RDF-star (RDF 1.2) and SPARQL-star, verified during planning (oxigraph 0.5.9,
`RdfFormat.supports_rdf_star`). It is a compiled (wheel) dependency; no external
service. Reserved alternative (`rdflib` + manual reification, pure-Python) is
recorded but not chosen — rdflib lacks RDF-star.
