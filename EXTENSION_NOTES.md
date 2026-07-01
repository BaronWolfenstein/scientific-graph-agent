# Extension Notes

This file records exactly what was changed from the original `scientific-graph-agent`
by Lina Faik. I adapted and instrumented the existing framework — I did not author it.

## Changes made (2026-06-24)

### 1. Anthropic backend (Step 1)
- Added `langchain-anthropic>=0.3.0` and `anthropic>=0.40.0` to `pyproject.toml`
- Created `src/agent_graph/llm.py`: `get_llm(temperature)` factory returning
  `ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000, ...)`
- Replaced all `ChatOpenAI(...)` instantiations in `nodes.py` (~7 sites) and
  `state.py` (`summarize_messages` reducer) with `get_llm(...)`
- API key read from `ANTHROPIC_API_KEY` env var; never hard-coded

### 2. PubMed retrieval (Step 2)
- Added `_fetch_pubmed_results` helper and `search_pubmed` LangChain `@tool`
  to `src/agent_graph/tools.py`
- Uses NCBI E-utilities (`esearch.fcgi` + `efetch.fcgi`) via stdlib `urllib` —
  no extra dependencies
- Returns the same paper-dict contract as the ArXiv tool
  (`id`, `title`, `authors`, `summary`, `url`, `published`, `source`) plus `pmid`
- Added `pubmed_researcher_node` to `nodes.py` and `"pubmed"` to `TOOL_REGISTRY`
- ArXiv tool and nodes remain intact; PubMed is additive

### 3. Two-audience structured-output node with schema-failure retry (Step 3)
- Created `src/agent_graph/schemas.py`: `Evidence`, `ClinicianSummary`,
  `TechnicalSummary` Pydantic v2 models
- Added `_extract_json` (strips markdown fences) and `_generate_with_retry`
  to `nodes.py`: prompts model, validates with `model_validate_json()`,
  re-prompts once with the validation error on failure (the retry is demonstrable
  and tested in `tests/test_schemas.py`)
- Added `dual_audience_node` to `nodes.py`: generates both summaries with
  grounding constraint (only cite PMIDs from retrieved papers)
- Added `clinician_summary` and `technical_summary` fields to `OutputState`
  and `InternalState` in `state.py`

### 4. HITL approval gate (Step 4)
- Added `hitl_approval_node` and `route_after_hitl` to `nodes.py`
- Uses LangGraph's `interrupt()` inside the node — graph suspends mid-node,
  resumes when `Command(resume={"action": "approve"|"reject"})` is passed
- Pattern follows the existing `approver_node` in the repo
- Gate fires after `dual_audience_node`, before summaries are finalized

### 5. Patched checkpointer (Steps 1, 4)
- Bumped `langgraph-checkpoint-sqlite` to `>=3.0.1` in `pyproject.toml`
- Added `langgraph-checkpoint>=4.0.1`
- Previous pin (`>=3.0.0`) had a known SQLi→RCE chain in older releases
- **Note on Python 3.9 environment:** The spec required `langgraph-checkpoint-sqlite>=3.0.1`
  and `langgraph-checkpoint>=4.0.1`, but those versions require Python 3.10+. The actual
  installed versions on this Python 3.9 system are `langgraph-checkpoint 2.1.2` and
  `langgraph-checkpoint-sqlite 2.0.11` — the latest 2.x series available for Python 3.9.
  These are fully functional for HITL interrupt/resume patterns used here.

### 6. Demo graph and entry point (Step 4)
- Added `create_demo_graph()` to `graph.py`:
  `clarifier → pubmed_researcher → summarizer → dual_audience → hitl_approval → END`
- Created `run_demo.py`: CLI entry point, handles interrupt/resume loop
- Example query: CAR-T cell therapy in relapsed/refractory DLBCL

### 7. Pre-HITL validation gate (2026-06-30)
- Added `validate_output_node` + `route_after_validation` to `nodes.py`, spliced
  between `dual_audience` and `hitl_approval` in `create_demo_graph`:
  `... → dual_audience → validate_output → {regenerate → dual_audience | approve → hitl_approval}`
- Two-layer machine check BEFORE the human gate, so the reviewer adjudicates
  *meaning*, not *form*:
  1. **Structural** — JSON Schema (`ClinicianSummary`/`TechnicalSummary.model_json_schema()`)
     via the `jsonschema` lib (added to `pyproject.toml`).
  2. **Grounding** — every cited PMID must be in the retrieved set (a cross-document
     constraint JSON Schema cannot express); catches hallucinated citations.
  3. **PMID identity** (`_pmid_conflicts`) — one PMID = one paper: catches the
     LIGER/Seurat *collision* (a PMID cited with conflicting sources) and
     *mis-attribution* (a cited PMID's source disagrees with the retrieved record).
     Deterministic, no extra API; the same invariant the KG reserves as
     FUNCTIONAL(pmid).
- Invalid drafts loop back to regenerate, bounded by `max_iterations` (gate bumps
  `iteration`); added `validation_errors` field to `InternalState`.
- Tested in `tests/test_validate_output.py` (7 tests: valid, structural, grounding,
  PMID collision, PMID mis-attribution, loop-break, wiring).
- **Why JSON Schema and not JSONB/JSON-LD:** JSON Schema *validates*; the other two
  solve adjacent, later problems and are reserved (see KG design spec §9):
  - **JSONB** — persist *approved* summaries in Postgres for audit/query
    ("every summary citing pmid X"); the after-approval storage layer, pairs with
    the reserved FHIR AuditEvent item.
  - **JSON-LD** — give the summary an `@context` mapping entities to RDF +
    LOINC/SNOMED/RxNorm URIs so an approved summary becomes a node in the reserved
    RDF-star knowledge graph; the semantic-interop bridge.

### 8. Offline GEPA prompt optimization (2026-06-30, decoupled)
- New `src/agent_graph/optimize/` package — offline tooling, NOT imported by the
  live LangGraph pipeline:
  - `metric.py` — `summarizer_metric(gold, pred, ...)` returning
    `dspy.Prediction(score, feedback)`: hard gates (JSON Schema + citation
    grounding, same invariants as the pre-HITL gate) × quality judges
    (`compute_faithfulness` + `compute_answer_relevance`, reused from `eval/`).
    Judges are injectable → deterministic parts unit-tested without API.
  - `program.py` — `DualAudienceProgram` with **two** signatures
    (`GenerateClinician`/`GenerateTechnical`) matching the two `dual_audience_node`
    SystemMessages, so each evolved instruction deploys 1:1.
  - `run_gepa.py` — offline `compile_program(trainset)` wiring `dspy.GEPA`; consumes
    API budget, not run by tests or the pipeline.
  - `harvest_and_optimize.py` — ready-to-run: regenerates `(query, papers)` examples
    from demo runs across families (oncology, cardiology/metabolic, neurology,
    **pathology**), splits train/val, prints the two evolved instructions.
- Deploy target: the guidance is extracted to swappable constants
  `CLINICIAN_GUIDANCE` / `TECHNICAL_GUIDANCE` in `nodes.py` (composed by
  `_audience_system_prompt`); the grounding rule + JSON schema stay fixed. Paste an
  evolved instruction into the matching constant.
- Added `dspy>=3.2` to `pyproject.toml`. Tests: `test_gepa_metric.py` (5),
  `test_gepa_program.py` (4), `test_dual_audience_prompt.py` (1).
- Design: `docs/superpowers/specs/2026-06-30-summarizer-gepa-optimization-design.md`
  (§6.1 = query-family generalization: one prompt vs family-conditioned;
  measure-per-family, shard only if divergent).
- Fully decoupled from the knowledge graph; uses only signals that exist in the
  built demo. HITL reject-reason capture + label trainset are a documented follow-up.

## What was NOT changed
- ArXiv and Wikipedia tools and nodes (intact, still usable)
- `create_graph`, `create_streaming_graph`, `create_graph_with_approval`,
  `create_map_reduce_graph` — all original graph factories unchanged
- `state.py` reducers (`keep_top_papers`, `take_max`) — unchanged
- Notebook demos — not updated (use original OpenAI key)
