# Quantitative Evidence Integration Plan

> **STATUS: ON HOLD (2026-07-01) — do not execute.** Blocked on the v1 domain KG (`src/agent_graph/kg/`) landing first. Phases 1–3 + Tasks 4b/7b are pure and could run early, but the decision (2026-07-01) is to hold the *entire* plan until v1 KG is merged, then resume from Phase 1. Re-confirm the effect-size engine's home is SGA (not causal_bench) before starting Phase 4.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Staging note:** This plan integrates a quantitative meta-analysis engine (the "causal notebook") into SGA's domain KG. Phases 1–2 are pure functions with NO dependency on the unbuilt `kg/` store and are fully bite-sized here. Phases 3–5 depend on the v1 KG (`rdfstar_store.py`, `graph_builder_node`) existing first; their tasks are specified at interface level with real signatures, and their per-step TDD is authored at execution time against the real v1 files (per the SGA spec §9 prohibition on TDD steps against unbuilt code). **Do not start Phase 3 until v1 KG is merged.**

**Goal:** Give SGA claims an optional quantitative effect-size payload (OR/RR/HR/logistic-β/RD + variance), pooled across papers by inverse-variance meta-analysis, with heterogeneous study reporting normalized to a common basis — pooling same-direction evidence and flagging MUTEX-opposed edges as contested.

**Architecture:** Lift the notebook's two backend-agnostic assets — the `DerivationRule` statistics engine and the Normal-Normal (inverse-variance) update — into new pure modules `kg/stats_derive.py` and additions to `kg/confidence.py`. Effect payloads ride as **reifier annotations on the existing claim triple** (one edge class; direction encoded in the predicate). The notebook's hand-rolled RDF\* I/O is discarded; its vocabulary maps onto SGA's pyoxigraph RDF-1.2 reifier model.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy.stats, pydantic, pyoxigraph 0.5.x (RDF 1.2 reifier), pytest.

## Global Constraints

- Effect sizes are a statement about a claim → modeled as reifier annotations, never a second edge class (see Edge Semantics Decision below). Verbatim from spec §4.
- `MUTEX = {(increases_risk_of, decreases_risk_of)}`; flag-not-reject — both edges persist, marked `contested`. Verbatim from spec §5.
- `FUNCTIONAL = {}` in v1; detection mechanism retained. Verbatim from spec §5.
- No network in tests. Verbatim from spec §11.
- All new stat/effect math is pure functions in `kg/stats_derive.py` / `kg/confidence.py` — no store, no LLM, no pandas-in-the-store-layer leakage.
- pyoxigraph is the only RDF\* substrate; NO hand-rolled RDF\* tokenizer/parser/serializer is ported from the notebook.

## Out of Scope (deferred, with declared home)

- **Concept→specific-variable mapping (`EXPERT_GPT_CAUSAL_MAP`) and multi-source edge merge (PC / expert / GPT-green-yellow).** This is entity/synonym **reconciliation** — SGA spec §9.1's reserved workstream, which is ordered *first* among the clinical plans and has its own brainstorm + spec. It does NOT ride in this plan. When the reconciliation spec is authored, the notebook's `EXPERT_GPT_CAUSAL_MAP` and source-weight scheme are its concrete starting inputs. Recorded here so the omission is a decision, not a silent drop.

---

## Edge Semantics Decision (recommended, adopt before Phase 3)

**Recommendation: one edge class — the claim triple — with quantitative effect as reifier annotations.**

An effect-bearing claim is a normal SGA base triple `(subject, predicate, object)` where:

- **Direction is the predicate.** A protective effect is `decreases_risk_of`; a harmful effect is `increases_risk_of`; a null/undirected association is `associated_with`. The *sign of the pooled effect* and the *predicate* must agree (enforced in Phase 4).
- **Magnitude is a reifier annotation bundle.** The reifier `r` gains: `kg:effect_measure` (OR/RR/HR/logB/RD/binary_rate), `kg:effect` (the pooled log-effect or risk-difference), `kg:effect_var` (pooled variance), `kg:effect_ci_low/high`, plus the derived-stat provenance from Phase 1. These are *more annotation triples on the same reifier* SGA already builds — additive, per spec §4.

**Why one class, not two:**

1. SGA's entire substrate is "annotations on a base triple." Effect size *is* a statement about a statement — the textbook reifier case. A second edge class would fork `query()`'s BFS, `to_context()`, the merge reducer, and serialization.
2. It maps the notebook 1:1: notebook `predictor→outcome` edge = SGA `subject→object` claim; notebook per-edge `Model`+`effect`+`variance` = SGA reifier annotations; notebook `IS_ESTIMATED_*` = reifier provenance flags.
3. **Confidence becomes two-track on the same reifier, not two graphs.** The existing Beta-Bernoulli *qualitative* confidence (how many papers, how strongly they agree) is unchanged and always present. When ≥1 paper supplies an effect size, the reifier *additionally* carries a pooled Normal posterior (mean, var). A claim can therefore answer both "do studies agree it's harmful?" (Beta-Bernoulli) and "how harmful, pooled?" (Normal). Neither track blocks the other; a claim with no numbers just omits the effect bundle.
4. **MUTEX integration is free.** `increases_risk_of` vs `decreases_risk_of` on the same `(subject, object)` pair is already the declared MUTEX pair. Same-direction papers pool into one reifier's Normal posterior; an opposite-direction paper lands on the *other* predicate's triple, and `_flag_conflicts` marks both `contested` exactly as today. Pooling and flagging compose without special-casing.

**Consequence for extraction:** the extractor keeps emitting a directed predicate; a new optional `effect` sub-object on `ScientificTriplet` carries the raw reported statistic. Direction disagreement between the reported sign and the chosen predicate is a Phase 4 validation, not an extraction concern.

---

## File Structure

- `src/agent_graph/kg/stats_derive.py` — **new, pure.** Port of the notebook's `DerivationRule` + `infer_statistics`. Normalizes heterogeneous reporting (counts / rates / CIs / p-values / SEs / study-quality) into a canonical `(effect, variance)` per `EffectMeasure`. No store, no LLM.
- `src/agent_graph/kg/confidence.py` — **extend.** Keep the existing Beta-Bernoulli functions. Add: `EffectMeasure` enum, `pool_normal()` (inverse-variance), `beta_binomial_update()` (generalizes vote-based Beta-Bernoulli to weighted event counts), `effect_direction()`, `effect_ci()`.
- `src/agent_graph/kg/ontology.py` — **extend.** Add `EffectMeasure` re-export and the `kg:effect_*` annotation URIs. MUTEX unchanged.
- `src/agent_graph/kg/extract.py` — **extend.** Add optional `ReportedEffect` sub-model to `ScientificTriplet`.
- `src/agent_graph/kg/rdfstar_store.py` — **extend.** `add_relation` accepts an optional `effect` dict; writes effect-bundle annotations onto the reifier; `query`/`to_context` surface pooled effect + CI.
- `src/agent_graph/kg/vocab_map.py` — **new.** The static map from notebook RDF\* keys/effect-type names → SGA reifier URIs / `EffectMeasure` (replaces the notebook's `RDF_TO_*` dicts; import-only tables, no parser).
- `tests/kg/test_stats_derive.py`, `tests/kg/test_confidence_effect.py`, `tests/kg/test_effect_store.py` — **new.**

---

## Phase 1 — Statistics derivation engine (pure, no v1 dependency)

Port the notebook's `DerivationRule`/`infer_statistics` into `kg/stats_derive.py`, trimmed to the canonical goal: from whatever a study reports, produce `(effect, variance)` in the log/RD basis, plus an `is_estimated` flag. Drop the notebook's pandas-DataFrame-wide design in favor of a per-record dataclass (SGA passes one evidence dict at a time, not a frame).

### Task 1: EffectMeasure + canonical record

**Files:**
- Create: `src/agent_graph/kg/stats_derive.py`
- Test: `tests/kg/test_stats_derive.py`

**Interfaces:**
- Produces: `class EffectMeasure(str, Enum)` with members `BINARY_RATE, OR, RR, HR, LOG_B, RD`; `@dataclass StudyStat` with fields `measure: EffectMeasure`, and optional `events, non_events, sample_size, event_rate, group_a_events, group_a_non_events, group_b_events, group_b_non_events, group_a_rate, group_b_rate, effect, ci_low, ci_high, p_value, std_error, study_quality` (all `float | None`), plus `weight: float = 1.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/kg/test_stats_derive.py
from agent_graph.kg.stats_derive import EffectMeasure, StudyStat

def test_studystat_defaults_and_measure():
    s = StudyStat(measure=EffectMeasure.OR, group_a_events=80, group_a_non_events=20,
                  group_b_events=61, group_b_non_events=41)
    assert s.measure is EffectMeasure.OR
    assert s.weight == 1.0
    assert s.effect is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_stats_derive.py::test_studystat_defaults_and_measure -v`
Expected: FAIL with "No module named 'agent_graph.kg.stats_derive'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/agent_graph/kg/stats_derive.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class EffectMeasure(str, Enum):
    BINARY_RATE = "binary_rate"
    OR = "odds_ratio"
    RR = "relative_risk"
    HR = "hazard_ratio"
    LOG_B = "logistic_beta"
    RD = "risk_difference"

@dataclass
class StudyStat:
    measure: EffectMeasure
    events: float | None = None
    non_events: float | None = None
    sample_size: float | None = None
    event_rate: float | None = None
    group_a_events: float | None = None
    group_a_non_events: float | None = None
    group_b_events: float | None = None
    group_b_non_events: float | None = None
    group_a_rate: float | None = None
    group_b_rate: float | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    std_error: float | None = None
    study_quality: str | None = None
    weight: float = 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_stats_derive.py::test_studystat_defaults_and_measure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/kg/stats_derive.py tests/kg/test_stats_derive.py
git commit -m "feat(kg): EffectMeasure enum and StudyStat record"
```

### Task 2: Comparative counts → (log-effect, variance)

**Files:**
- Modify: `src/agent_graph/kg/stats_derive.py`
- Test: `tests/kg/test_stats_derive.py`

**Interfaces:**
- Consumes: `StudyStat`, `EffectMeasure` (Task 1).
- Produces: `derive_effect(s: StudyStat) -> tuple[float, float, bool]` returning `(effect, variance, is_estimated)` in the canonical basis — log(OR/RR/HR) or raw (RD, logistic-β). Raises `ValueError` if insufficient data.

- [ ] **Step 1: Write the failing test**

```python
import math
from agent_graph.kg.stats_derive import StudyStat, EffectMeasure, derive_effect

def test_or_from_two_group_counts():
    # OR = (80/20)/(61/41) = 4.0 / 1.4878 = 2.6885 ; logOR = 0.9892
    s = StudyStat(measure=EffectMeasure.OR, group_a_events=80, group_a_non_events=20,
                  group_b_events=61, group_b_non_events=41)
    effect, var, est = derive_effect(s)
    assert math.isclose(effect, math.log((80/20)/(61/41)), rel_tol=1e-9)
    # var(logOR) = 1/80 + 1/20 + 1/61 + 1/41
    assert math.isclose(var, 1/80 + 1/20 + 1/61 + 1/41, rel_tol=1e-9)
    assert est is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_stats_derive.py::test_or_from_two_group_counts -v`
Expected: FAIL with "cannot import name 'derive_effect'"

- [ ] **Step 3: Write minimal implementation**

```python
import math

_QUALITY_TO_N = {"uncertain": 25, "weak": 25, "moderate": 50, "strong": 100}

def _counts_from_group(events, non_events, rate, quality) -> tuple[float, float, bool]:
    """Return (events, non_events, is_estimated) for one arm."""
    if events is not None and non_events is not None:
        return events, non_events, False
    if rate is not None:
        n = _QUALITY_TO_N.get((quality or "").lower(), 50)
        return n * rate, n * (1 - rate), True
    raise ValueError("insufficient arm data")

def derive_effect(s: StudyStat) -> tuple[float, float, bool]:
    if s.measure is EffectMeasure.OR:
        ae, an, ea = _counts_from_group(s.group_a_events, s.group_a_non_events, s.group_a_rate, s.study_quality)
        be, bn, eb = _counts_from_group(s.group_b_events, s.group_b_non_events, s.group_b_rate, s.study_quality)
        odds_a, odds_b = ae / an, be / bn
        effect = math.log(odds_a / odds_b)
        var = 1/ae + 1/an + 1/be + 1/bn
        return effect, var, (ea or eb)
    raise ValueError(f"unsupported measure for counts path: {s.measure}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_stats_derive.py::test_or_from_two_group_counts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): OR from two-group counts with estimated-N fallback"
```

### Task 3: RR and RD from counts; measure dispatch

**Files:**
- Modify: `src/agent_graph/kg/stats_derive.py`
- Test: `tests/kg/test_stats_derive.py`

**Interfaces:**
- Consumes: `derive_effect`, `_counts_from_group` (Task 2).
- Produces: `derive_effect` extended to `RR` (log-RR, `var = 1/ae - 1/na + 1/be - 1/nb`) and `RD` (raw difference, `var = pa(1-pa)/na + pb(1-pb)/nb`).

- [ ] **Step 1: Write the failing test**

```python
def test_rr_from_counts():
    s = StudyStat(measure=EffectMeasure.RR, group_a_events=80, group_a_non_events=20,
                  group_b_events=61, group_b_non_events=41)
    effect, var, est = derive_effect(s)
    pa, pb = 80/100, 61/102
    assert math.isclose(effect, math.log(pa/pb), rel_tol=1e-9)
    assert math.isclose(var, 1/80 - 1/100 + 1/61 - 1/102, rel_tol=1e-9)

def test_rd_from_counts():
    s = StudyStat(measure=EffectMeasure.RD, group_a_events=80, group_a_non_events=20,
                  group_b_events=61, group_b_non_events=41)
    effect, var, est = derive_effect(s)
    pa, pb, na, nb = 0.8, 61/102, 100, 102
    assert math.isclose(effect, pa - pb, rel_tol=1e-9)
    assert math.isclose(var, pa*(1-pa)/na + pb*(1-pb)/nb, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_stats_derive.py -k "rr_from_counts or rd_from_counts" -v`
Expected: FAIL with "unsupported measure for counts path"

- [ ] **Step 3: Write minimal implementation**

```python
    if s.measure is EffectMeasure.RR:
        ae, an, ea = _counts_from_group(s.group_a_events, s.group_a_non_events, s.group_a_rate, s.study_quality)
        be, bn, eb = _counts_from_group(s.group_b_events, s.group_b_non_events, s.group_b_rate, s.study_quality)
        na, nb = ae + an, be + bn
        pa, pb = ae / na, be / nb
        return math.log(pa / pb), 1/ae - 1/na + 1/be - 1/nb, (ea or eb)
    if s.measure is EffectMeasure.RD:
        ae, an, ea = _counts_from_group(s.group_a_events, s.group_a_non_events, s.group_a_rate, s.study_quality)
        be, bn, eb = _counts_from_group(s.group_b_events, s.group_b_non_events, s.group_b_rate, s.study_quality)
        na, nb = ae + an, be + bn
        pa, pb = ae / na, be / nb
        return pa - pb, pa*(1-pa)/na + pb*(1-pb)/nb, (ea or eb)
```

(Insert before the final `raise ValueError` in `derive_effect`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_stats_derive.py -k "rr_from_counts or rd_from_counts" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): RR and RD from counts"
```

### Task 4: Effect-value + variance path (from reported effect / CI / p-value / SE)

**Files:**
- Modify: `src/agent_graph/kg/stats_derive.py`
- Test: `tests/kg/test_stats_derive.py`

**Interfaces:**
- Consumes: `derive_effect`, `StudyStat`, `EffectMeasure`.
- Produces: within `derive_effect`, a value-path branch used when arm counts are absent but a summary effect is reported. For ratio measures (OR/RR/HR) input `effect`/`ci_*` are on the ratio scale and are log-transformed; for `LOG_B`/`RD` they are already on the analysis scale. Variance is derived from (in priority) explicit `std_error²`, else the CI half-width `((log_hi - log_lo)/2 / 1.96)²`, else `(|log_effect| / z_p)²` from the p-value. `HR` and `LOG_B` are value-path-only (no counts).

- [ ] **Step 1: Write the failing test**

```python
from scipy.stats import norm

def test_hr_from_value_and_ci():
    # HR=2.0, 95% CI [1.5, 2.667] symmetric in log space around log(2)
    s = StudyStat(measure=EffectMeasure.HR, effect=2.0, ci_low=1.5, ci_high=2.6667)
    effect, var, est = derive_effect(s)
    assert math.isclose(effect, math.log(2.0), rel_tol=1e-6)
    se = (math.log(2.6667) - math.log(1.5)) / 2 / norm.ppf(0.975)
    assert math.isclose(var, se**2, rel_tol=1e-3)
    assert est is True  # summary-level, no raw counts

def test_rd_from_value_and_se():
    s = StudyStat(measure=EffectMeasure.RD, effect=0.12, std_error=0.05)
    effect, var, est = derive_effect(s)
    assert math.isclose(effect, 0.12, rel_tol=1e-9)
    assert math.isclose(var, 0.05**2, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_stats_derive.py -k "hr_from_value or rd_from_value" -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
from scipy.stats import norm

_RATIO = {EffectMeasure.OR, EffectMeasure.RR, EffectMeasure.HR}

def _value_path(s: StudyStat) -> tuple[float, float, bool]:
    if s.effect is None:
        raise ValueError("no reported effect for value path")
    ratio = s.measure in _RATIO
    effect = math.log(s.effect) if ratio else s.effect
    lo = math.log(s.ci_low) if (ratio and s.ci_low is not None) else s.ci_low
    hi = math.log(s.ci_high) if (ratio and s.ci_high is not None) else s.ci_high
    if s.std_error is not None:
        var = s.std_error ** 2
    elif lo is not None and hi is not None:
        se = (hi - lo) / 2 / norm.ppf(0.975)
        var = se ** 2
    elif s.p_value is not None:
        z = norm.ppf(1 - s.p_value / 2)
        var = (abs(effect) / z) ** 2
    else:
        raise ValueError("no variance source (need SE, CI, or p-value)")
    return effect, var, True
```

Then, near the top of `derive_effect`, add a guard so measures without counts, or records lacking arm data, use the value path:

```python
def derive_effect(s: StudyStat) -> tuple[float, float, bool]:
    if s.measure in (EffectMeasure.HR, EffectMeasure.LOG_B) or s.group_a_events is s.group_a_rate is None:
        if s.effect is not None:
            return _value_path(s)
    # ... existing OR/RR/RD counts branches ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_stats_derive.py -k "hr_from_value or rd_from_value" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): value-path effect/variance from CI, SE, or p-value"
```

### Task 4b: Binary single-arm derivation (event rate / counts → weighted events, non-events)

Closes absorb #1's binary path — the notebook's `BINARY_FROM_RATE` chain. This is what feeds the Beta-Binomial track (Task 7); without it a binary-rate-only study can't contribute.

**Files:**
- Modify: `src/agent_graph/kg/stats_derive.py`
- Test: `tests/kg/test_stats_derive.py`

**Interfaces:**
- Consumes: `StudyStat`, `EffectMeasure`, `_QUALITY_TO_N` (Task 2).
- Produces: `derive_binary(s: StudyStat) -> tuple[float, float, bool]` returning `(events, non_events, is_estimated)`. Priority: explicit `events`/`non_events` (measured); else `event_rate` × sample size, where sample size is `sample_size` if given, else `study_quality`→`_QUALITY_TO_N`, else default 50 (estimated). Raises `ValueError` if neither counts nor rate present.

- [ ] **Step 1: Write the failing test**

```python
from agent_graph.kg.stats_derive import derive_binary

def test_binary_from_explicit_counts():
    s = StudyStat(measure=EffectMeasure.BINARY_RATE, events=50, non_events=70)
    ev, nev, est = derive_binary(s)
    assert (ev, nev, est) == (50, 70, False)

def test_binary_from_rate_and_sample_size():
    s = StudyStat(measure=EffectMeasure.BINARY_RATE, event_rate=0.6, sample_size=100)
    ev, nev, est = derive_binary(s)
    assert math.isclose(ev, 60.0) and math.isclose(nev, 40.0) and est is True

def test_binary_from_rate_and_quality():
    s = StudyStat(measure=EffectMeasure.BINARY_RATE, event_rate=0.6, study_quality="strong")
    ev, nev, est = derive_binary(s)   # strong → N=100
    assert math.isclose(ev, 60.0) and math.isclose(nev, 40.0) and est is True

def test_binary_from_rate_default_n():
    s = StudyStat(measure=EffectMeasure.BINARY_RATE, event_rate=0.6)
    ev, nev, est = derive_binary(s)   # default N=50
    assert math.isclose(ev, 30.0) and math.isclose(nev, 20.0) and est is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_stats_derive.py -k binary_from -v`
Expected: FAIL with "cannot import name 'derive_binary'"

- [ ] **Step 3: Write minimal implementation**

```python
def derive_binary(s: StudyStat) -> tuple[float, float, bool]:
    """Return (events, non_events, is_estimated) for a binary single-arm study."""
    if s.events is not None and s.non_events is not None:
        return s.events, s.non_events, False
    if s.event_rate is not None:
        if s.sample_size is not None:
            n = s.sample_size
        else:
            n = _QUALITY_TO_N.get((s.study_quality or "").lower(), 50)
        return n * s.event_rate, n * (1 - s.event_rate), True
    raise ValueError("binary study needs counts or an event rate")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_stats_derive.py -k binary_from -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): binary single-arm derivation from counts, rate, or quality"
```

---

## Phase 2 — Pooling & confidence (pure, extends confidence.py)

### Task 5: Normal-Normal inverse-variance pooling

**Files:**
- Modify: `src/agent_graph/kg/confidence.py`
- Test: `tests/kg/test_confidence_effect.py`

**Interfaces:**
- Consumes: nothing new (pure numpy/math).
- Produces: `pool_normal(prior_mean: float, prior_var: float, samples: list[tuple[float, float, float]]) -> tuple[float, float]` where each sample is `(mean, variance, weight)`; returns `(posterior_mean, posterior_var)` by weighted precision. Weakly-informative prior `pool_normal(0.0, 1e6, ...)` reproduces a fixed-effect meta-analysis.

- [ ] **Step 1: Write the failing test**

```python
# tests/kg/test_confidence_effect.py
import math
from agent_graph.kg.confidence import pool_normal

def test_pool_two_studies_fixed_effect():
    # weakly-informative prior; two studies, equal weight
    post_mean, post_var = pool_normal(0.0, 1e6, [(0.2, 0.04, 1.0), (0.4, 0.01, 1.0)])
    # precision-weighted: (0.2/0.04 + 0.4/0.01) / (1/0.04 + 1/0.01)
    expected = (0.2/0.04 + 0.4/0.01) / (1/0.04 + 1/0.01)
    assert math.isclose(post_mean, expected, rel_tol=1e-4)
    assert post_var < 0.01  # tighter than either study

def test_weight_downscales_precision():
    _, v_full = pool_normal(0.0, 1e6, [(0.3, 0.02, 1.0)])
    _, v_half = pool_normal(0.0, 1e6, [(0.3, 0.02, 0.5)])
    assert v_half > v_full  # a half-weighted study contributes less precision
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_confidence_effect.py -k pool -v`
Expected: FAIL with "cannot import name 'pool_normal'"

- [ ] **Step 3: Write minimal implementation**

```python
def pool_normal(prior_mean: float, prior_var: float,
                samples: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Inverse-variance (precision-weighted) Normal-Normal update.
    Each sample is (mean, variance, weight). Weight scales the sample's precision."""
    prior_prec = 1.0 / prior_var
    post_prec = prior_prec
    prec_weighted_mean_sum = prior_prec * prior_mean
    for mean, var, weight in samples:
        prec = weight / var
        post_prec += prec
        prec_weighted_mean_sum += mean * prec
    post_var = 1.0 / post_prec
    return post_var * prec_weighted_mean_sum, post_var
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_confidence_effect.py -k pool -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): Normal-Normal inverse-variance pooling"
```

### Task 6: Effect direction + CI readout

**Files:**
- Modify: `src/agent_graph/kg/confidence.py`
- Test: `tests/kg/test_confidence_effect.py`

**Interfaces:**
- Consumes: `pool_normal` (Task 5); `EffectMeasure` (imported from `stats_derive`).
- Produces: `effect_direction(measure, effect) -> Literal["increase", "decrease", "null"]` (ratio measures: null at log-effect 0; RD/logistic-β: null at 0); `effect_ci(mean, var, measure) -> tuple[float, float]` returning the 95% CI back-transformed to the ratio scale for ratio measures, raw for RD/logistic-β.

- [ ] **Step 1: Write the failing test**

```python
from agent_graph.kg.stats_derive import EffectMeasure
from agent_graph.kg.confidence import effect_direction, effect_ci

def test_direction_ratio_and_rd():
    assert effect_direction(EffectMeasure.OR, 0.5) == "increase"   # logOR>0 → OR>1
    assert effect_direction(EffectMeasure.OR, -0.5) == "decrease"
    assert effect_direction(EffectMeasure.RD, 0.0) == "null"

def test_ci_backtransform_for_ratio():
    lo, hi = effect_ci(math.log(2.0), (0.1)**2, EffectMeasure.OR)
    assert lo < 2.0 < hi
    assert math.isclose(lo, math.exp(math.log(2.0) - 1.96*0.1), rel_tol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_confidence_effect.py -k "direction or backtransform" -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
import math
from typing import Literal
from agent_graph.kg.stats_derive import EffectMeasure

_RATIO = {EffectMeasure.OR, EffectMeasure.RR, EffectMeasure.HR}

def effect_direction(measure: EffectMeasure, effect: float,
                     tol: float = 1e-9) -> Literal["increase", "decrease", "null"]:
    if abs(effect) <= tol:
        return "null"
    return "increase" if effect > 0 else "decrease"

def effect_ci(mean: float, var: float, measure: EffectMeasure) -> tuple[float, float]:
    half = 1.96 * math.sqrt(var)
    lo, hi = mean - half, mean + half
    if measure in _RATIO:
        return math.exp(lo), math.exp(hi)
    return lo, hi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_confidence_effect.py -k "direction or backtransform" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): effect direction and CI back-transform"
```

### Task 7: Beta-Binomial reconciliation (generalize the vote-based Beta-Bernoulli)

**Files:**
- Modify: `src/agent_graph/kg/confidence.py`
- Test: `tests/kg/test_confidence_effect.py`

**Interfaces:**
- Consumes: existing `PRIOR_A`, `PRIOR_B` from `confidence.py`.
- Produces: `beta_binomial_update(alpha, beta, samples: list[tuple[float, float, float]]) -> tuple[float, float]` where each sample is `(events, non_events, weight)`; returns `(alpha', beta')`. The existing support/refute vote path is shown to be the special case `events=1, non_events=0` (support) / `events=0, non_events=1` (refute).

- [ ] **Step 1: Write the failing test**

```python
from agent_graph.kg.confidence import beta_binomial_update

def test_beta_binomial_weighted_counts():
    a, b = beta_binomial_update(1.0, 1.0, [(8.0, 2.0, 0.5), (6.0, 4.0, 1.0)])
    assert math.isclose(a, 1.0 + 0.5*8 + 1.0*6, rel_tol=1e-9)
    assert math.isclose(b, 1.0 + 0.5*2 + 1.0*4, rel_tol=1e-9)

def test_vote_is_special_case():
    a, b = beta_binomial_update(1.0, 1.0, [(1.0, 0.0, 0.8), (0.0, 1.0, 0.8)])
    assert math.isclose(a, 1.8) and math.isclose(b, 1.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_confidence_effect.py -k beta_binomial -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def beta_binomial_update(alpha: float, beta: float,
                         samples: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Weighted Beta-Binomial update. Each sample is (events, non_events, weight).
    Vote-based Beta-Bernoulli is the special case events/non_events in {0,1}."""
    for events, non_events, weight in samples:
        alpha += weight * events
        beta += weight * non_events
    return alpha, beta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_confidence_effect.py -k beta_binomial -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): weighted Beta-Binomial generalizing vote-based update"
```

### Task 7b: Estimated-evidence discount (make `is_estimated` do work)

Closes absorb #4 — turns the carried `is_estimated` flag into an actual precision/weight discount rather than inert provenance. This is the Leak-A-aligned move: imputed evidence contributes less than measured evidence. `graph_builder_node` (Task 12) multiplies the base relevance-weight by this factor before pooling.

**Files:**
- Modify: `src/agent_graph/kg/confidence.py`
- Test: `tests/kg/test_confidence_effect.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `EST_DISCOUNT: float = 0.5` module constant and `effective_weight(base_weight: float, is_estimated: bool, discount: float = EST_DISCOUNT) -> float` returning `base_weight * discount` when estimated, else `base_weight`. Feeds the `weight` slot of both `pool_normal` samples and `beta_binomial_update` samples, so an imputed study contributes proportionally less precision on the Normal track and fewer pseudo-counts on the Beta-Binomial track.

- [ ] **Step 1: Write the failing test**

```python
from agent_graph.kg.confidence import effective_weight, EST_DISCOUNT, pool_normal

def test_effective_weight_discounts_estimated():
    assert effective_weight(0.8, is_estimated=False) == 0.8
    assert math.isclose(effective_weight(0.8, is_estimated=True), 0.8 * EST_DISCOUNT)

def test_estimated_study_pools_looser():
    # same effect/variance, one measured one estimated → estimated yields higher posterior var
    _, v_measured = pool_normal(0.0, 1e6, [(0.3, 0.02, effective_weight(1.0, False))])
    _, v_estimated = pool_normal(0.0, 1e6, [(0.3, 0.02, effective_weight(1.0, True))])
    assert v_estimated > v_measured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_confidence_effect.py -k "effective_weight or pools_looser" -v`
Expected: FAIL with "cannot import name 'effective_weight'"

- [ ] **Step 3: Write minimal implementation**

```python
EST_DISCOUNT = 0.5

def effective_weight(base_weight: float, is_estimated: bool,
                     discount: float = EST_DISCOUNT) -> float:
    """Downscale an imputed study's contribution. Leak-A-aligned: estimated
    evidence contributes less precision (Normal) / fewer pseudo-counts (Beta-Binomial)."""
    return base_weight * discount if is_estimated else base_weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_confidence_effect.py -k "effective_weight or pools_looser" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): estimated-evidence precision discount"
```

---

## Phase 3 — Vocabulary map (pure tables; no v1 store dependency, but consumed by Phase 4)

### Task 8: Notebook vocab → SGA reifier URIs / EffectMeasure

**Files:**
- Create: `src/agent_graph/kg/vocab_map.py`
- Test: `tests/kg/test_vocab_map.py`

**Interfaces:**
- Consumes: `EffectMeasure` (Task 1).
- Produces: `EFFECT_TYPE_TO_MEASURE: dict[str, EffectMeasure]` mapping the notebook's `ComparativeLogOR`/`BinaryRate`/… names to `EffectMeasure`; `EFFECT_ANNOTATION_URIS: dict[str, str]` mapping `effect`, `effect_var`, `effect_ci_low`, `effect_ci_high`, `effect_measure`, `is_estimated` to `kg:`-prefixed URI strings. Import-only tables; NO parser, NO tokenizer.

- [ ] **Step 1: Write the failing test**

```python
# tests/kg/test_vocab_map.py
from agent_graph.kg.vocab_map import EFFECT_TYPE_TO_MEASURE, EFFECT_ANNOTATION_URIS
from agent_graph.kg.stats_derive import EffectMeasure

def test_effect_type_names_map():
    assert EFFECT_TYPE_TO_MEASURE["ComparativeLogOR"] is EffectMeasure.OR
    assert EFFECT_TYPE_TO_MEASURE["BinaryRate"] is EffectMeasure.BINARY_RATE

def test_annotation_uris_are_kg_prefixed():
    assert EFFECT_ANNOTATION_URIS["effect"].startswith("http://") or ":" in EFFECT_ANNOTATION_URIS["effect"]
    assert set(EFFECT_ANNOTATION_URIS) >= {"effect", "effect_var", "effect_measure", "is_estimated"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kg/test_vocab_map.py -v`
Expected: FAIL with "No module named 'agent_graph.kg.vocab_map'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/agent_graph/kg/vocab_map.py
from agent_graph.kg.stats_derive import EffectMeasure

EFFECT_TYPE_TO_MEASURE = {
    "BinaryRate": EffectMeasure.BINARY_RATE,
    "ComparativeLogOR": EffectMeasure.OR,
    "ComparativeLogRR": EffectMeasure.RR,
    "ComparativeLogHR": EffectMeasure.HR,
    "ComparativeLogB": EffectMeasure.LOG_B,
    "ComparativeRD": EffectMeasure.RD,
}

_KG = "https://sga.example/kg#"
EFFECT_ANNOTATION_URIS = {
    "effect_measure": _KG + "effect_measure",
    "effect": _KG + "effect",
    "effect_var": _KG + "effect_var",
    "effect_ci_low": _KG + "effect_ci_low",
    "effect_ci_high": _KG + "effect_ci_high",
    "is_estimated": _KG + "is_estimated",
}
```

(At execution time, replace `_KG` base with the actual namespace `ontology.py` mints, and re-export from `ontology.py` per the module structure.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/kg/test_vocab_map.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(kg): notebook-vocabulary → SGA reifier URI map"
```

---

## Phase 4 — Store integration (DEPENDS ON v1 `rdfstar_store.py`; author TDD at execution)

> These tasks require the v1 `KnowledgeGraph` Protocol and `rdfstar_store.py` to exist. Interface-level specs are given; per-step failing tests are authored at execution time against the real store, following the v1 test patterns in `tests/kg/test_rdfstar_store.py`. Do NOT invent store method signatures ahead of v1.

### Task 9 (interface): `add_relation` accepts an optional effect payload

- **Files:** Modify `src/agent_graph/kg/rdfstar_store.py`; extend `tests/kg/test_effect_store.py`.
- **Interfaces — Produces:** `add_relation(..., effect: dict | None = None)`. When `effect` is present it is a `StudyStat`-shaped dict; the store calls `derive_effect` (Phase 1) to get `(effect, var, is_estimated)`, then pools it into the reifier's running Normal posterior via `pool_normal` (Phase 5 helper), writing `kg:effect_measure/effect/effect_var/effect_ci_low/effect_ci_high/is_estimated` annotation triples on the SAME reifier that already carries `kg:confidence`. The qualitative Beta-Bernoulli track is updated in parallel and unchanged.
- **Direction/predicate guard:** if `effect_direction(measure, pooled_effect)` disagrees with the triple's predicate (`increases_risk_of` vs `decreases_risk_of`), the store routes the evidence to the correctly-directed triple and lets `_flag_conflicts` mark the MUTEX pair `contested` — pooling same-direction, flagging opposed, per Global Constraints.
- **Deliverable test themes (author against v1):** (a) a single OR study writes an effect bundle whose back-transformed CI brackets the point estimate; (b) two same-direction studies pool to a tighter variance than either; (c) an opposite-direction study creates the MUTEX-opposed triple and both are flagged `contested`; (d) a claim with no effect writes no `kg:effect_*` triples and its Beta-Bernoulli confidence is byte-identical to pre-change.

### Task 10 (interface): `query` and `to_context` surface pooled effect

- **Files:** Modify `src/agent_graph/kg/rdfstar_store.py`.
- **Interfaces — Produces:** `query(...)` result dicts gain optional keys `effect_measure`, `effect`, `effect_ci`, `is_estimated`, `contested`. `to_context(...)` renders effect-bearing claims as e.g. `imatinib —increases_risk_of→ cardiotoxicity (OR 1.8, 95% CI 1.2–2.7; pooled 3 studies) ⚠CONTESTED`.
- **Deliverable test themes:** ranked query returns effect fields when present and omits them cleanly when absent; `to_context` string contains the back-transformed CI and the `⚠CONTESTED` marker only for flagged pairs.

---

## Phase 5 — Extraction + node wiring (DEPENDS ON v1 `extract.py` + `graph_builder_node`)

### Task 11 (interface): optional `ReportedEffect` on `ScientificTriplet`

- **Files:** Modify `src/agent_graph/kg/extract.py`.
- **Interfaces — Produces:** `class ReportedEffect(BaseModel)` with `measure: Literal["odds_ratio","relative_risk","hazard_ratio","logistic_beta","risk_difference","binary_rate"]` and the optional reported fields (`effect`, `ci_low`, `ci_high`, `p_value`, `std_error`, arm counts/rates). `ScientificTriplet` gains `effect: ReportedEffect | None = None`. The prompt instructs the extractor to fill `effect` ONLY when the abstract reports a quantitative statistic, and to choose the directed predicate consistent with the effect's sign.
- **Deliverable test themes:** structured-output round-trips a triple with and without an effect; an abstract snippet reporting "OR 2.1 (95% CI 1.3–3.4)" yields a `ReportedEffect` with `measure=odds_ratio`.

### Task 12 (interface): `graph_builder_node` passes effect through

- **Files:** Modify `src/agent_graph/kg/` node wiring (the `graph_builder_node` added by v1).
- **Interfaces — Produces:** when a `ScientificTriplet.effect` is present, the node builds the `StudyStat`-shaped dict (mapping `relevance`→`weight` as today) and passes it as `add_relation(..., effect=...)`. The store applies `effective_weight(base_weight, is_estimated)` (Task 7b) after `derive_effect`/`derive_binary` reports `is_estimated`, so imputed studies pool looser. Extraction/derivation failure on the effect is caught and the claim is still added qualitatively — best-effort, never blocks the pipeline (matches v1 §10).
- **Deliverable test themes:** stubbed LLM returning a triplet-with-effect adds both the qualitative vote and the effect bundle; a stubbed triplet whose effect fails `derive_effect` still adds the qualitative claim and logs a skip.

---

## Self-Review

**Absorb / conflict coverage (the four + three this plan owes):**
- Absorb 1 — DerivationRule stats engine → Phase 1 Tasks 1–4 (comparative) + **Task 4b (binary single-arm)**. Complete for the six measures.
- Absorb 2 — Normal-Normal inverse-variance → Task 5. Complete.
- Absorb 3 — concept→variable mapping / multi-source merge → **explicitly Out of Scope**, handed to the reconciliation workstream (spec §9.1). Decision recorded, not silent.
- Absorb 4 — estimated-vs-measured → `is_estimated` derived (Tasks 2–4b), **consumed as a precision discount (Task 7b)**, written as provenance (Task 9). Now does work, not inert.
- Conflict A — RDF\* strategy → Task 8 + Global Constraints (pyoxigraph only, no ported parser). Resolved.
- Conflict B — pool vs flag → Task 9 direction guard. Resolved.
- Conflict C — edge semantics → Edge Semantics Decision, enforced Task 9. Resolved.

**Other spec coverage:** Beta-Binomial reconciliation → Task 7. Vocab mapping → Task 8 + Phase 4. Effect surfaced to generation → Tasks 10, 12.

**Placeholder scan:** Phases 1–3 contain complete code and exact commands. Phases 4–5 are deliberately interface-level with named signatures and "deliverable test themes" rather than fake bite-sized steps, because their target files do not exist until v1 KG merges — writing invented `rdfstar_store.py` line numbers now would be the exact placeholder failure the SGA spec §9 warns against. This is flagged at the top and per-phase.

**Type consistency:** `EffectMeasure` defined in Task 1, imported by Tasks 5–8. `derive_effect` signature `(StudyStat) -> (float, float, bool)` consistent Tasks 2–4, consumed Task 9. `pool_normal`/`beta_binomial_update` sample tuple shapes: Normal is `(mean, var, weight)`, Beta-Binomial is `(events, non_events, weight)` — distinct and used consistently.
