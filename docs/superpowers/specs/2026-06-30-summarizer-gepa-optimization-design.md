# Summarizer GEPA Optimization — Design

**Date:** 2026-06-30
**Status:** Approved, building
**Component:** `src/agent_graph/optimize/` (new package)
**Decoupled from:** the domain knowledge graph (spec 2026-06-27). Shares no code
or dependency with the KG; can be built, run, and shipped independently.

## 1. Goal

Use **GEPA** (reflective prompt evolution, arXiv 2507.19457) via **DSPy** to
optimize the summary-generation prompt *offline*, using signals we already have:
the pre-HITL validation gate (schema + citation grounding) and the existing
LLM-as-judge evaluators (faithfulness, answer-relevance) — plus, optionally,
accumulated HITL approve/reject labels. The optimized prompt is swapped back into
the live node; the live pipeline is otherwise unchanged.

GEPA is chosen because its strength is **natural-language feedback**, and our gate
and judges already emit exactly that (schema error strings, ungrounded-PMID lists,
judge reasons) — not just scalars.

## 2. Why this is decoupled from the KG

Every signal exists in the *built* demo agent today:
- schema validity + citation grounding — the pre-HITL gate (`nodes.validate_output_node`);
- faithfulness — `eval/faithfulness.py::compute_faithfulness`;
- answer-relevance — `eval/answer_relevance.py::compute_answer_relevance`;
- human preference — HITL approve/reject.

None of these touch the RDF-star KG or Beta-Bernoulli confidence. This is the
quality/optimization track running parallel to the provenance/graph track.

## 3. What GEPA optimizes, and the degenerate-summary trap

GEPA maximizes the metric. If the metric were *only* the hard gates
(schema-valid + grounded), GEPA could evolve **degenerate** summaries that pass
trivially (cite nothing, minimize claims). The metric therefore combines:

- **Hard gates (correctness floor):** schema validity → score 0 on failure;
  citation grounding → heavy multiplicative penalty when a cited PMID was not
  retrieved.
- **Quality (usefulness):** `0.5 * faithfulness + 0.5 * answer_relevance`.
  `answer_relevance` is what defeats the degenerate case — a summary that says
  nothing scores low on answering the query.

GEPA's Pareto search handles this multi-objective tension; the feedback string
concatenates the diagnostic from each layer.

## 4. Architecture

```
src/agent_graph/optimize/
├── __init__.py
├── metric.py     # summarizer_metric(gold, pred, ...) -> dspy.Prediction(score, feedback)
│                 #   hard gates (schema+grounding) + quality judges (injectable)
├── program.py    # DSPy Signature + Module wrapping dual-audience generation
└── run_gepa.py   # OFFLINE compile script: wires dspy.GEPA(metric) + trainset -> optimized prompt
```

- **`metric.py`** is the reusable core and the only fully unit-tested piece. The
  two judges are **injected** (default to the real `compute_faithfulness` /
  `compute_answer_relevance`) so the deterministic gate logic is testable without
  API calls.
- **`program.py`** wraps only the generation step as a DSPy module; the rest of the
  LangGraph pipeline is untouched. Blast radius = one node's prompt.
- **`run_gepa.py`** is an **offline** step (needs an Anthropic budget + a training
  set); it is wired and importable but NOT executed by tests or the live pipeline.

## 5. Metric contract

```python
summarizer_metric(gold, pred, trace=None, pred_name=None, pred_trace=None,
                  *, faithfulness_fn=compute_faithfulness,
                     relevance_fn=compute_answer_relevance) -> dspy.Prediction
```
- `gold`: has `.query: str`, `.papers: list[dict]` (each with `pmid`).
- `pred`: has `.clinician_summary: dict`, `.technical_summary: dict`.
- Returns `dspy.Prediction(score: float in [0,1], feedback: str)`.

Scoring:
1. schema-invalid → `score=0.0`, feedback = the jsonschema error(s).
2. else compute `ungrounded` = cited PMIDs not in retrieved set;
   `quality = 0.5*faithfulness + 0.5*answer_relevance`;
   `score = quality * (1.0 if not ungrounded else 0.3)`.
3. feedback always concatenates: grounding issues (which PMIDs), faithfulness
   fraction, answer-relevance reason — the diagnostic GEPA reflects on.

## 6. Training data

- **Now (automatic):** harvest `(query, papers)` from logged pipeline states /
  demo runs. GEPA is sample-efficient (~35× fewer rollouts than GRPO), so a modest
  set suffices. The metric's judges provide the reward with no human labels.
- **Later (human preference):** fold HITL approve/reject into the trainset as
  positive/negative examples. Capturing a free-text **reject reason** (small
  enhancement to `hitl_approval_node` / `run_demo.py`) turns each rejection into a
  textual GEPA feedback datum — GEPA's strongest signal. This build ships the
  automatic-signal metric only; HITL reject-reason capture + trainset integration
  is a documented follow-up (§10). The automatic gate + judges are sufficient to
  begin optimizing.

### 6.1 Query-family generalization

GEPA optimizes ONE prompt against the training distribution — an average-case
optimum. Whether a single prompt suffices depends on whether query families
(oncology-therapy, cardiology/metabolic outcomes, neurology, **pathology**
[diagnostic/computational/molecular], epidemiology, …) diverge in what "good"
looks like. Three things make one prompt viable *here*:

- **The metric is family-agnostic.** Schema + grounding + faithfulness +
  answer-relevance are the same invariants across every specialty. What varies by
  family is content/emphasis, not the correctness objective — and the metric only
  scores the objective. So the evolved instruction's transferable core (cite only
  retrieved PMIDs, one PMID = one paper, structure, hedge when thin) generalizes.
- **GEPA can evolve *conditional* instructions** ("for interventional questions
  emphasize N/endpoints; for diagnostic-accuracy questions emphasize
  sensitivity/specificity and the reference standard") when the trainset spans
  families, rather than a bland compromise.
- **Prompts generalize across families far better than fine-tuned weights**, so
  prompt-level optimization is the right tool for a single cross-family artifact.

**Decision rule (measure, don't guess):**
1. Train ONE prompt on a deliberately diverse seed set spanning all families
   (`SEED_QUERIES` includes oncology, cardiology/metabolic, neurology, and three
   pathology queries). Scale to ~20–40 (several per family) — 6–9 overfits those
   specific queries.
2. Evaluate on a **held-out valset** with per-family breakdown.
3. If per-family scores are uniform → ship the single prompt. **Only if a family
   lags materially** → go family-conditioned: cluster queries into families, GEPA
   per cluster, and route at inference (classify query → family → select prompt).
   That router + prompt-per-family registry is real machinery — do NOT build it
   until the measurement justifies it (YAGNI).

**Deploy target (per §7):** the two evolved instructions map 1:1 onto the live
node's guidance because the DSPy program has two signatures
(`GenerateClinician`/`GenerateTechnical`) matching the two `dual_audience_node`
SystemMessages. Deploy by editing the `CLINICIAN_GUIDANCE` / `TECHNICAL_GUIDANCE`
constants in `nodes.py`; the grounding rule + JSON schema stay fixed.

## 7. Offline compile / deploy loop

`run_gepa.py`:
1. Build `dspy.LM` pointed at Claude (same model as `get_llm`).
2. Construct the `program.py` module + the trainset of `dspy.Example(query, papers)`.
3. `optimized = dspy.GEPA(metric=summarizer_metric, reflection_lm=<claude>,
   auto="light").compile(program, trainset=..., valset=...)`.
4. Export the optimized instruction/prompt; a human reviews it, then it is pasted
   into `dual_audience_node` (or loaded as a prompt asset). **Held-out valset**
   confirms generalization, not metric overfit.

This is periodic and offline — no change to live latency or graph structure.

## 8. Testing

- `metric.py`: full TDD with **stubbed judges** — schema-invalid → 0 + schema
  feedback; ungrounded citation → penalized + names the PMID; clean + high judges →
  high score; feedback always diagnostic. No network.
- `program.py` / `run_gepa.py`: import/construct smoke test only (real GEPA run is
  offline, API-bound).

## 9. Dependencies

Add `dspy>=3.2` to `pyproject.toml` (pulls litellm etc.). No change to the live
runtime path — DSPy is only imported by the `optimize/` package and the offline
script.

## 10. Reserved / out of scope

- Executing the GEPA optimization run (offline, user-triggered with budget + data).
- Converting the whole graph to DSPy (only the one generation node is wrapped).
- HITL reject-reason capture + HITL-label trainset integration (documented
  follow-up; not built in this iteration — automatic signals suffice to start).
- Antislop-style phrasing suppression — a *different* quality axis that composes
  with this, not part of it.
