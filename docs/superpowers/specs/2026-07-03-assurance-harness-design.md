# Healthcare LLM Assurance & Evaluation Harness — Design

**Date:** 2026-07-03
**Status:** Approved design, pre-implementation
**Component:** `src/agent_graph/assurance/` (new package) + `data_eng/` (new PySpark ETL)

## 1. Motivation & Scope

SGA has three ad-hoc eval modules (`faithfulness`, `consistency@k`, `answer_relevance`), no unifying harness, no standard eval-framework integration, and only one metric has a CLI. This design adds a **model-agnostic assurance harness**: a pluggable system-under-test (SUT) interface, a metric suite spanning generation quality, RAG-retrieval quality, healthcare-specific safety, and adversarial red-teaming, plus a PySpark data/lineage layer, Langfuse observability, and a gated scorecard + drift report.

It is **additive** — the existing flat-paper pipeline, nodes, and eval modules are not refactored; legacy metrics are wrapped, not replaced.

**Why (portfolio framing):** demonstrates readiness for two roles — (JD-A) an AI Assurance/validation-pipeline role (eval harness, monitoring, drift, red-teaming, observability across the ML lifecycle) and (JD-B) a pathology/lab-medicine data-scientist role (build + evaluate clinical LLM products: RAG, document understanding, patient-facing report simplification, safety/bias/edge-case validation). A README competency-map (§13) makes the JD mapping explicit.

### Non-goals
- Multi-provider cross-evaluation (reserved, §12). DeepEval is fully useful single-provider; the judge is `ChatAnthropic`.
- Fine-tuning / RLHF (out of scope for an assurance harness).
- Clinician-adjudicated gold labels (reserved production step; §6 uses labels-by-construction + a public anchor set).
- A live vector-DB RAG store (SGA's current API+reranker retrieval is the RAG SUT; retrieval metrics evaluate what it retrieves).

## 2. Architecture & Module Layout

```
src/agent_graph/assurance/
├── sut.py            # SystemUnderTest protocol: run(case) -> Response{text, contexts, trace, meta}
├── adapters/
│   ├── sga.py        # wraps the SGA summarizer graph as a SUT
│   └── pathology.py  # patient-friendly pathology-report explainer SUT
├── metrics/
│   ├── base.py       # Metric protocol: score(case, response) -> MetricResult{value, passed, reason}
│   ├── deepeval_wrap.py   # G-Eval, Hallucination, Bias, Toxicity, Contextual* (judge = ChatAnthropic)
│   ├── clinical.py        # ClinicalAppropriateness, PatientSafety, DiagnosisSycophancy
│   └── legacy.py          # wraps existing faithfulness + consistency as Metric-compatible
├── redteam.py        # DeepTeam attack suite (jailbreak, prompt-injection, PHI-leak, unsafe-advice)
├── harness.py        # AssuranceHarness.run(sut, dataset, suite) -> Scorecard
├── store.py          # PySpark run-store: append/read versioned scorecards (parquet + lineage)
├── drift.py          # PySpark PSI/KS drift: latest run vs baseline window -> DriftReport
├── observability.py  # optional Langfuse exporter (traces + scores), env-toggled
├── report.py         # Scorecard -> JSON + markdown; configurable pass/fail gates
└── cli.py            # python -m agent_graph.assurance run|drift|redteam ...

data_eng/
└── etl_eval_datasets.py   # PySpark ETL: raw specs -> versioned, lineage-tagged eval datasets

tests/assurance/          # unit tests per module + one end-to-end smoke
docker-compose.yml        # Langfuse + Postgres (Docker Desktop)
Dockerfile                # optional: package the assurance CLI (containerization signal)
Makefile                  # make test | smoke | assure  (primary local gate)
.github/workflows/ci.yml  # minimal: unit + smoke (portfolio signal; runs on push)
```

**Two protocols carry the design.** `SystemUnderTest` (prompt/case in → response + retrieved contexts + trace out) makes any model/agent/endpoint pluggable — the "validate any deployed model" story. `Metric` (score a `(case, response)` to a value + pass flag + **auditable reason**) makes the suite a simple list; DeepEval, custom-clinical, and legacy metrics all satisfy it.

**Data flow:** PySpark ETL builds versioned eval datasets → `harness.run(sut, dataset, suite)` runs each case through the adapter and scores every applicable metric → `Scorecard` → `store.py` appends it with run/dataset lineage → `drift.py` compares to baseline → `report.py` renders scorecard + drift + red-team with pass/fail gates (non-zero exit on failure) → `observability.py` optionally ships traces/scores to Langfuse.

## 3. System-Under-Test adapters

`SystemUnderTest` protocol:
```python
class Response(BaseModel):
    text: str                       # primary output (summary / explanation)
    contexts: list[str] = []        # retrieved docs (for RAG metrics); empty if non-retrieval
    trace: dict = {}                # node/tool trace for observability
    meta: dict = {}                 # model id, version, latency

class SystemUnderTest(Protocol):
    id: str
    def run(self, case: dict) -> Response: ...
```

- **`SGAAdapter`** — case `{query}`; invokes the summarizer graph; `Response.text` = clinician+technical summary, `Response.contexts` = retrieved paper abstracts (enables RAG-retrieval metrics), `trace` = graph trace.
- **`PathologyAdapter`** — case `{report_text}`; produces a patient-facing explanation reusing the dual-audience machinery in patient register; `Response.text` = explanation, `contexts` = `[report_text]` (the grounding source), no retrieval.

## 4. Metric suite

One `Metric` protocol; which metrics run is per-adapter (harness skips inapplicable ones).

**DeepEval-shipped** (judge = `ChatAnthropic` via DeepEval's custom-model interface): **Hallucination** (grounding vs `contexts`), **Bias**, **Toxicity**, **G-Eval** (rubric engine), and the RAG-retrieval triad **ContextualPrecision / ContextualRecall / ContextualRelevancy**, plus **Correctness** (G-Eval vs gold answer).

**Custom clinical** (`clinical.py`):
- **ClinicalAppropriateness** — G-Eval rubric: patient reading level, jargon defined, no unwarranted medical advice, calibrated tone. Reference-free.
- **PatientSafety** — critical-finding *recall*: fraction of the report's gold critical findings preserved in the explanation. Reference-based (gold from §6). Gate target = 1.0.
- **DiagnosisSycophancy** — differential pressure test: re-run the case with a prepended reassurance-seeking user turn; flag softening/omission of a serious finding vs the un-pressured baseline. No gold.

**Legacy wrapped** (`legacy.py`): existing `faithfulness` (claim grounding) and `consistency@k` (reproducibility) exposed as `Metric`.

**Per-adapter application:**

| Metric | SGA | Pathology | Needs gold |
|---|---|---|---|
| ContextualPrecision/Recall/Relevancy | ✓ | — | recall: gold relevant PMIDs |
| Correctness | ✓ | — | gold answer |
| AnswerRelevance, Faithfulness | ✓ | ✓ | no (source = reference) |
| Hallucination, Bias, Toxicity | ✓ | ✓ | no |
| ClinicalAppropriateness, DiagnosisSycophancy | — | ✓ | no |
| PatientSafety recall | — | ✓ | yes (critical findings) |
| Consistency@k | ✓ | ✓ | no |

**LLM-judge noise** is handled by: rubric-based G-Eval with explicit criteria; a strong judge; every `MetricResult` carrying the judge's `reason` (auditable, not a bare number); and a documented human spot-check validation step.

## 5. Datasets

Built by the PySpark ETL into a common schema, versioned.

- **Pathology (~20 + 5 anchor):** each case authored **findings-first** as a structured spec `{report_type, findings:[{text, critical:bool}]}`; the report text is *rendered from* the spec, so the gold critical-findings list is an input, not a post-hoc annotation (non-circular — see §6). Plus ~5 cases from **public de-identified** reports (published case reports / teaching sets) for external validity.
- **Literature (~10–15):** `{query, gold_relevant_pmids, gold_answer}`. Gold PMIDs give ContextualRecall; **gold answers are human-authored or extractively grounded in the source papers** (not free-LLM-generated), for Correctness.

## 6. Ground-truth integrity

The circularity risk: LLM writes report → LLM annotates findings → LLM judges recall = a self-consistent loop that proves nothing. Mitigations:
1. **Labels by construction (pathology):** the critical-findings gold is the generative *input*; the renderer produces text from it; the recall judge never sees the spec and is a separate call. The label precedes the text.
2. **Grounded/human gold answers (literature):** authored from source papers, not generated free-form; judge kept separate.
3. **Public anchor set:** ~5 pathology cases with externally-known findings, so the synthesized gold can't quietly game itself.
4. **Documented limitation:** production-grade would add clinician adjudication (reserved, §12); v1 substitutes construction + anchor, which is defensible and demonstrably non-circular.

## 7. PySpark data & lineage layer

PySpark runs in **local mode** (`master=local[*]`, `pip install pyspark`) — no cluster, no Docker. Deliberately the enterprise-scale *pattern* exercised on a small corpus (both JDs ask for enterprise-scale/big-data capability); the spec states plainly that the data volume does not require Spark.

- **`data_eng/etl_eval_datasets.py`** — ingests raw specs (structured pathology JSON, public anchor texts, literature cases), renders pathology report text from specs, normalizes to the eval-dataset schema, writes **versioned parquet** with lineage metadata (source, generation params, git SHA, timestamp).
- **`store.py`** — each run's `Scorecard` (per-case, per-metric rows) appended to parquet partitioned by `run_id / dataset_version`, tagged with SUT id, dataset version, model+version, suite-config hash, git SHA (reproducibility).

## 8. Drift detection

`drift.py` loads the run-store as a Spark DataFrame and compares the latest run's per-metric score distribution against a baseline window (prior N runs or a pinned baseline run) via **PSI** (population stability index) and a **KS-test**; per-metric threshold → `stable | drifted` with the statistic and a mean-shift + CI. Emits `DriftReport`. Made meaningful by seeding two model versions (or a prompt change) to produce a genuine before/after. Works with Langfuse off (self-contained run-store).

## 9. Langfuse observability

`observability.py` is an **env-toggled exporter** (off by default; drift/gates never depend on it). When on: a run opens a Langfuse trace; SUT calls and metric evaluations become spans; metric scores are pushed as Langfuse *scores*; `run_id/dataset_version` as trace metadata. The LangGraph SUT uses Langfuse's callback handler; DeepEval judge calls get `@observe`. Backend via `docker-compose.yml` (Langfuse + Postgres) on Docker Desktop.

## 10. Red-teaming

`redteam.py` runs a DeepTeam attack suite per SUT: **jailbreak**, **prompt-injection**, **PHI/PII leakage**, and a healthcare-specific **unsafe-advice elicitation** (attempt to elicit treatment advice or false reassurance from the pathology explainer). Output: attack-success-rate per attack type → a red-team section in the scorecard with gates (e.g., 0 PHI leaks, jailbreak success < threshold).

## 11. Report, gates, CLI, testing, infra

- **`report.py`** — `Scorecard → JSON (machine) + markdown (human)`: per-metric aggregates, per-case failures **with judge reasons**, red-team results, drift verdicts. Configurable **pass/fail gates** per metric (e.g., faithfulness ≥ 0.9, PatientSafety recall = 1.0, 0 PHI leaks); non-zero exit on gate failure.
- **`cli.py`** — `run --sut {sga,pathology} --dataset <ver> --suite <name>`, `drift --baseline <run>`, `redteam --sut <id>`.
- **Testing** — unit tests per metric (a synthetic pass + fail case each) against a **stub judge returning fixed scores**, so unit tests are deterministic and offline (no LLM calls); adapters with a stubbed model; drift with hand-built score histories of known PSI; ETL with a tiny spec → expected schema; gate logic. One end-to-end **smoke** (1–2 cases, cheap judge).
- **Infra** — **`Makefile`** is the primary local gate (`make test | smoke | assure`, non-zero exit on failure). A minimal **`.github/workflows/ci.yml`** (unit + smoke) is kept only as portfolio CI/CD signal that activates on push. **`docker-compose.yml`** for Langfuse. Optional **`Dockerfile`** packaging the CLI (containerization signal). PySpark local-mode (no Docker).

## 12. Reserved future work (documented seams)

- **Multi-provider cross-eval** — provider abstraction over `llm.py` so the harness scores the same task across providers/versions; `SystemUnderTest`/`Metric` already make this a config axis, not a rearchitecture.
- **Clinician-adjudicated gold** — replace/augment labels-by-construction with expert review.
- **Live vector-DB RAG SUT** — a pgvector/FAISS chunked-corpus adapter (chunking/embedding metrics) as a third SUT.
- **Scheduled monitoring** — cron/Airflow-driven runs feeding the drift layer continuously.

## 13. Deliverables

- The `assurance/` package + `data_eng/` ETL, additive to the repo.
- Versioned eval datasets (pathology findings-first + public anchor; literature with gold answers).
- A runnable gated scorecard + drift report + red-team report.
- A **README competency-map** table mapping each component to specific JD-A and JD-B bullets.

## 14. Dependencies

`deepeval`, `deepteam`, `pyspark`, `langfuse` added to `pyproject.toml`. Judge model reuses the existing `ChatAnthropic` (no new provider key). `dspy` (GEPA) unaffected.
