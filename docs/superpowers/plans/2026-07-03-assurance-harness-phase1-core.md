# Assurance Harness — Phase 1 (Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working, end-to-end assurance harness core — pluggable `SystemUnderTest` and `Metric` protocols, SGA + pathology adapters, the existing faithfulness metric wrapped, a harness runner, and a gated JSON/markdown scorecard with a CLI — using only existing dependencies.

**Architecture:** New additive package `src/agent_graph/assurance/`. Two protocols (`SystemUnderTest`, `Metric`) decouple what's evaluated from how it's scored. Adapters wrap the existing SGA graph and a new patient-report explainer via dependency injection (a graph/LLM can be injected for offline tests). The harness runs each case through the SUT, scores every applicable metric, and emits a `Scorecard` the report layer renders and gates.

**Tech Stack:** Python ≥3.9, pydantic, langchain-core, the existing `agent_graph` modules (`llm.get_llm`, `graph.create_graph`, `eval.faithfulness`). No new external dependencies in Phase 1 (DeepEval, PySpark, Langfuse, DeepTeam arrive in Phases 2–4).

## Global Constraints

- Additive only: do not modify `reranker.py`, `nodes.py`, `graph.py`, `eval/*`, or `schemas.py`. New code imports them.
- Judge/LLM reuse the existing `agent_graph.llm.get_llm` — no new provider key.
- Every `Metric.score` returns a `MetricResult` with an auditable `reason` string (never a bare number).
- Adapters accept an injected graph/LLM so unit tests run offline with no LLM/network calls.
- Python ≥3.9 (repo `requires-python`). Use `from __future__ import annotations` in new modules that use `X | Y` syntax.
- Commit after every task; message ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Run from repo root `/Users/noahrahman/git/scientific-graph-agent`. Tests via `python -m pytest`.
- Branch: `feat/assurance-harness` (already checked out; the spec is committed there).

---

### Task 1: Package scaffold + core protocols and models

**Files:**
- Create: `src/agent_graph/assurance/__init__.py`
- Create: `src/agent_graph/assurance/sut.py`
- Create: `src/agent_graph/assurance/metrics/__init__.py`
- Create: `src/agent_graph/assurance/metrics/base.py`
- Test: `tests/assurance/__init__.py`, `tests/assurance/test_core_models.py`

**Interfaces:**
- Produces:
  - `Response(text: str, contexts: list[str] = [], trace: dict = {}, meta: dict = {})` (pydantic).
  - `SystemUnderTest` Protocol: attr `id: str`; `run(case: dict) -> Response`.
  - `MetricResult(name: str, value: float, passed: bool, reason: str)` (pydantic).
  - `Metric` Protocol: attr `name: str`; `applies_to(sut_id: str) -> bool`; `score(case: dict, response: Response) -> MetricResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/assurance/__init__.py` (empty) and `tests/assurance/test_core_models.py`:

```python
from agent_graph.assurance.sut import Response
from agent_graph.assurance.metrics.base import MetricResult


def test_response_defaults():
    r = Response(text="hello")
    assert r.text == "hello"
    assert r.contexts == [] and r.trace == {} and r.meta == {}


def test_response_with_contexts():
    r = Response(text="x", contexts=["c1", "c2"])
    assert r.contexts == ["c1", "c2"]


def test_metric_result_fields():
    m = MetricResult(name="faithfulness", value=0.8, passed=False, reason="below 0.9")
    assert m.name == "faithfulness" and m.value == 0.8
    assert m.passed is False and "0.9" in m.reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assurance/test_core_models.py -q`
Expected: FAIL (ModuleNotFoundError: `agent_graph.assurance`).

- [ ] **Step 3: Write the modules**

Create `src/agent_graph/assurance/__init__.py`:
```python
"""Model-agnostic LLM assurance & evaluation harness (see docs spec 2026-07-03)."""
```

Create `src/agent_graph/assurance/sut.py`:
```python
"""System-under-test protocol and the response it returns."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Response(BaseModel):
    """One system-under-test output. `contexts` are retrieved docs (RAG metrics)
    or the grounding source; empty for pure-generation SUTs."""
    text: str
    contexts: list[str] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)


@runtime_checkable
class SystemUnderTest(Protocol):
    id: str
    def run(self, case: dict) -> Response: ...
```

Create `src/agent_graph/assurance/metrics/__init__.py` (empty).

Create `src/agent_graph/assurance/metrics/base.py`:
```python
"""Metric protocol and its result."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_graph.assurance.sut import Response


class MetricResult(BaseModel):
    name: str
    value: float
    passed: bool
    reason: str


@runtime_checkable
class Metric(Protocol):
    name: str
    def applies_to(self, sut_id: str) -> bool: ...
    def score(self, case: dict, response: Response) -> MetricResult: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assurance/test_core_models.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/assurance/__init__.py src/agent_graph/assurance/sut.py \
        src/agent_graph/assurance/metrics/__init__.py src/agent_graph/assurance/metrics/base.py \
        tests/assurance/__init__.py tests/assurance/test_core_models.py
git commit -m "feat(assurance): core SystemUnderTest/Metric protocols + models

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Faithfulness metric (legacy wrap)

**Files:**
- Create: `src/agent_graph/assurance/metrics/legacy.py`
- Test: `tests/assurance/test_legacy_metrics.py`

**Interfaces:**
- Consumes: `Response`, `MetricResult` (Task 1); `agent_graph.eval.faithfulness.compute_faithfulness_single(summary, papers, llm=None) -> (score, 0, 0)`.
- Produces: `FaithfulnessMetric(threshold: float = 0.9, llm=None)` with `name = "faithfulness"`, `applies_to(sut_id) -> True`, `score(case, response) -> MetricResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/assurance/test_legacy_metrics.py`:

```python
from agent_graph.assurance.sut import Response
from agent_graph.assurance.metrics.legacy import FaithfulnessMetric
from agent_graph.eval.faithfulness import FaithfulnessScore


class _FakeStructured:
    def __init__(self, score): self._score = score
    def invoke(self, messages):
        return FaithfulnessScore(supported_fraction=self._score, reasoning="stub")


class _FakeLLM:
    """Minimal stand-in for ChatAnthropic used by compute_faithfulness_single."""
    def __init__(self, score): self._score = score
    def with_structured_output(self, schema): return _FakeStructured(self._score)


def test_faithfulness_passes_above_threshold():
    m = FaithfulnessMetric(threshold=0.9, llm=_FakeLLM(0.95))
    r = m.score({"query": "q"}, Response(text="summary", contexts=["abstract text"]))
    assert r.name == "faithfulness"
    assert r.value == 0.95 and r.passed is True
    assert "0.95" in r.reason


def test_faithfulness_fails_below_threshold():
    m = FaithfulnessMetric(threshold=0.9, llm=_FakeLLM(0.5))
    r = m.score({"query": "q"}, Response(text="summary", contexts=["abstract"]))
    assert r.value == 0.5 and r.passed is False


def test_applies_to_any_sut():
    m = FaithfulnessMetric()
    assert m.applies_to("sga") and m.applies_to("pathology")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assurance/test_legacy_metrics.py -q`
Expected: FAIL (ImportError: cannot import `FaithfulnessMetric`).

- [ ] **Step 3: Write the module**

Create `src/agent_graph/assurance/metrics/legacy.py`:
```python
"""Existing SGA eval metrics exposed as assurance Metrics."""
from __future__ import annotations

from agent_graph.assurance.metrics.base import MetricResult
from agent_graph.assurance.sut import Response
from agent_graph.eval.faithfulness import compute_faithfulness_single


class FaithfulnessMetric:
    """Fraction of the output's claims grounded in `response.contexts`.

    Wraps the single-call faithfulness estimator; contexts are passed as the
    `papers` list it expects (it reads each item's 'summary'). llm is injectable
    for offline tests.
    """
    name = "faithfulness"

    def __init__(self, threshold: float = 0.9, llm=None):
        self.threshold = threshold
        self._llm = llm

    def applies_to(self, sut_id: str) -> bool:
        return True

    def score(self, case: dict, response: Response) -> MetricResult:
        papers = [{"summary": c} for c in response.contexts]
        value, _, _ = compute_faithfulness_single(response.text, papers, self._llm)
        return MetricResult(
            name=self.name,
            value=float(value),
            passed=value >= self.threshold,
            reason=f"supported_fraction={value:.2f} (threshold {self.threshold})",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assurance/test_legacy_metrics.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/assurance/metrics/legacy.py tests/assurance/test_legacy_metrics.py
git commit -m "feat(assurance): FaithfulnessMetric wrapping the existing eval

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: SGA and pathology adapters

**Files:**
- Create: `src/agent_graph/assurance/adapters/__init__.py`
- Create: `src/agent_graph/assurance/adapters/sga.py`
- Create: `src/agent_graph/assurance/adapters/pathology.py`
- Test: `tests/assurance/test_adapters.py`

**Interfaces:**
- Consumes: `Response` (Task 1). Real deps (injected in tests): `agent_graph.graph.create_graph`, `agent_graph.llm.get_llm`.
- Produces:
  - `SGAAdapter(graph=None)` — `id = "sga"`, `run(case)` invokes the compiled graph with `{"query": case["query"], "max_papers": case.get("max_papers", 4)}` and returns `Response(text=summary, contexts=[p["summary"] for p in papers])`.
  - `PathologyAdapter(llm=None)` — `id = "pathology"`, `run(case)` builds a patient-facing explanation of `case["report_text"]`; returns `Response(text=explanation, contexts=[case["report_text"]])`.

- [ ] **Step 1: Write the failing test**

Create `tests/assurance/test_adapters.py`:

```python
from agent_graph.assurance.adapters.sga import SGAAdapter
from agent_graph.assurance.adapters.pathology import PathologyAdapter


class _FakeGraph:
    def invoke(self, payload, config=None):
        assert payload["query"] == "crispr"
        return {
            "summary": "- CRISPR edits DNA [Paper 1]",
            "papers": [{"title": "T", "summary": "abstract about crispr", "url": "u"}],
            "clinician_summary": {"bottom_line": "bl"},
        }


class _FakeMsg:
    def __init__(self, content): self.content = content


class _FakeLLM:
    def invoke(self, messages):
        return _FakeMsg("Your biopsy shows a benign finding; nothing urgent.")


def test_sga_adapter_maps_summary_and_contexts():
    a = SGAAdapter(graph=_FakeGraph())
    assert a.id == "sga"
    r = a.run({"query": "crispr"})
    assert "CRISPR" in r.text
    assert r.contexts == ["abstract about crispr"]


def test_pathology_adapter_explains_and_grounds():
    a = PathologyAdapter(llm=_FakeLLM())
    assert a.id == "pathology"
    report = "Invasive ductal carcinoma, margins involved."
    r = a.run({"report_text": report})
    assert "benign" in r.text.lower()          # comes from the stub
    assert r.contexts == [report]              # grounding source retained
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assurance/test_adapters.py -q`
Expected: FAIL (ModuleNotFoundError: `agent_graph.assurance.adapters`).

- [ ] **Step 3: Write the modules**

Create `src/agent_graph/assurance/adapters/__init__.py` (empty).

Create `src/agent_graph/assurance/adapters/sga.py`:
```python
"""SGA literature-summarizer as a SystemUnderTest."""
from __future__ import annotations

from agent_graph.assurance.sut import Response


class SGAAdapter:
    id = "sga"

    def __init__(self, graph=None):
        # Injected compiled graph for tests; built lazily in production.
        self._graph = graph

    def _get_graph(self):
        if self._graph is None:
            from agent_graph.graph import create_graph
            self._graph = create_graph()
        return self._graph

    def run(self, case: dict) -> Response:
        graph = self._get_graph()
        payload = {"query": case["query"], "max_papers": case.get("max_papers", 4)}
        config = {"configurable": {"thread_id": case.get("thread_id", "assurance")}}
        result = graph.invoke(payload, config=config)
        papers = result.get("papers", []) or []
        return Response(
            text=result.get("summary", "") or "",
            contexts=[p.get("summary", "") for p in papers],
            trace={"n_papers": len(papers)},
            meta={"clinician_summary": result.get("clinician_summary")},
        )
```

Create `src/agent_graph/assurance/adapters/pathology.py`:
```python
"""Patient-friendly pathology-report explainer as a SystemUnderTest."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent_graph.assurance.sut import Response

_PATIENT_PROMPT = (
    "You are helping a patient and their family understand a pathology report. "
    "Rewrite it in plain language at about an 8th-grade reading level. "
    "Preserve every clinically important finding accurately — do not omit or soften "
    "serious findings, and do not add reassurance or medical advice that the report "
    "does not state. Define any medical term you must use."
)


class PathologyAdapter:
    id = "pathology"

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from agent_graph.llm import get_llm
            self._llm = get_llm(temperature=0, max_tokens=800)
        return self._llm

    def run(self, case: dict) -> Response:
        report = case["report_text"]
        llm = self._get_llm()
        out = llm.invoke([
            SystemMessage(content=_PATIENT_PROMPT),
            HumanMessage(content=report),
        ])
        return Response(text=out.content, contexts=[report])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assurance/test_adapters.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/assurance/adapters/ tests/assurance/test_adapters.py
git commit -m "feat(assurance): SGA and pathology SystemUnderTest adapters

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Harness runner + scorecard

**Files:**
- Create: `src/agent_graph/assurance/harness.py`
- Test: `tests/assurance/test_harness.py`

**Interfaces:**
- Consumes: `SystemUnderTest`, `Response` (Task 1); `Metric`, `MetricResult` (Task 1).
- Produces:
  - `CaseResult(case_id: str, metrics: list[MetricResult])` (pydantic).
  - `Scorecard(sut_id: str, dataset_id: str, results: list[CaseResult])` (pydantic).
  - `AssuranceHarness.run(sut, dataset: list[dict], metrics: list, dataset_id: str = "adhoc") -> Scorecard` — runs each case through `sut`, scores every metric whose `applies_to(sut.id)` is True; each case's id is `case.get("id", str(index))`.

- [ ] **Step 1: Write the failing test**

Create `tests/assurance/test_harness.py`:

```python
from agent_graph.assurance.sut import Response
from agent_graph.assurance.metrics.base import MetricResult
from agent_graph.assurance.harness import AssuranceHarness, Scorecard


class _StubSUT:
    id = "sga"
    def run(self, case):
        return Response(text=f"answer to {case['query']}", contexts=["ctx"])


class _AlwaysMetric:
    name = "always"
    def applies_to(self, sut_id): return True
    def score(self, case, response):
        return MetricResult(name="always", value=1.0, passed=True, reason="ok")


class _PathologyOnlyMetric:
    name = "patho_only"
    def applies_to(self, sut_id): return sut_id == "pathology"
    def score(self, case, response):
        return MetricResult(name="patho_only", value=0.0, passed=False, reason="n/a")


def test_harness_runs_applicable_metrics_only():
    sc = AssuranceHarness().run(
        _StubSUT(),
        dataset=[{"id": "c1", "query": "q1"}, {"id": "c2", "query": "q2"}],
        metrics=[_AlwaysMetric(), _PathologyOnlyMetric()],
        dataset_id="lit_v1",
    )
    assert isinstance(sc, Scorecard)
    assert sc.sut_id == "sga" and sc.dataset_id == "lit_v1"
    assert [c.case_id for c in sc.results] == ["c1", "c2"]
    # PathologyOnlyMetric must be skipped for the sga SUT
    assert [m.name for m in sc.results[0].metrics] == ["always"]


def test_case_id_defaults_to_index():
    sc = AssuranceHarness().run(_StubSUT(), [{"query": "q"}], [_AlwaysMetric()])
    assert sc.results[0].case_id == "0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assurance/test_harness.py -q`
Expected: FAIL (ImportError: `AssuranceHarness`).

- [ ] **Step 3: Write the module**

Create `src/agent_graph/assurance/harness.py`:
```python
"""The assurance harness: run a dataset through a SUT and score every metric."""
from __future__ import annotations

from pydantic import BaseModel

from agent_graph.assurance.metrics.base import MetricResult


class CaseResult(BaseModel):
    case_id: str
    metrics: list[MetricResult]


class Scorecard(BaseModel):
    sut_id: str
    dataset_id: str
    results: list[CaseResult]


class AssuranceHarness:
    def run(self, sut, dataset: list[dict], metrics: list,
            dataset_id: str = "adhoc") -> Scorecard:
        applicable = [m for m in metrics if m.applies_to(sut.id)]
        results: list[CaseResult] = []
        for i, case in enumerate(dataset):
            response = sut.run(case)
            case_id = str(case.get("id", i))
            results.append(CaseResult(
                case_id=case_id,
                metrics=[m.score(case, response) for m in applicable],
            ))
        return Scorecard(sut_id=sut.id, dataset_id=dataset_id, results=results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assurance/test_harness.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/assurance/harness.py tests/assurance/test_harness.py
git commit -m "feat(assurance): harness runner + Scorecard with per-SUT metric filtering

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Report + pass/fail gates

**Files:**
- Create: `src/agent_graph/assurance/report.py`
- Test: `tests/assurance/test_report.py`

**Interfaces:**
- Consumes: `Scorecard`, `CaseResult` (Task 4); `MetricResult` (Task 1).
- Produces:
  - `scorecard_to_dict(sc: Scorecard) -> dict` (JSON-ready).
  - `scorecard_to_markdown(sc: Scorecard) -> str`.
  - `apply_gates(sc: Scorecard) -> tuple[bool, list[str]]` — returns `(all_passed, failure_lines)`; a gate fails when any `MetricResult.passed` is False; each failure line is `"<case_id>/<metric>: <reason>"`.

- [ ] **Step 1: Write the failing test**

Create `tests/assurance/test_report.py`:

```python
from agent_graph.assurance.metrics.base import MetricResult
from agent_graph.assurance.harness import CaseResult, Scorecard
from agent_graph.assurance.report import (
    apply_gates, scorecard_to_dict, scorecard_to_markdown)


def _sc():
    return Scorecard(sut_id="pathology", dataset_id="patho_v1", results=[
        CaseResult(case_id="c1", metrics=[
            MetricResult(name="faithfulness", value=0.95, passed=True, reason="ok"),
            MetricResult(name="patient_safety", value=0.5, passed=False,
                         reason="dropped critical finding"),
        ]),
    ])


def test_apply_gates_flags_failures():
    passed, lines = apply_gates(_sc())
    assert passed is False
    assert lines == ["c1/patient_safety: dropped critical finding"]


def test_apply_gates_all_pass():
    sc = Scorecard(sut_id="sga", dataset_id="d", results=[
        CaseResult(case_id="c1", metrics=[
            MetricResult(name="faithfulness", value=1.0, passed=True, reason="ok")]),
    ])
    passed, lines = apply_gates(sc)
    assert passed is True and lines == []


def test_dict_and_markdown_render():
    d = scorecard_to_dict(_sc())
    assert d["sut_id"] == "pathology"
    assert d["results"][0]["metrics"][1]["name"] == "patient_safety"
    md = scorecard_to_markdown(_sc())
    assert "pathology" in md and "patient_safety" in md and "0.50" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assurance/test_report.py -q`
Expected: FAIL (ImportError: `agent_graph.assurance.report`).

- [ ] **Step 3: Write the module**

Create `src/agent_graph/assurance/report.py`:
```python
"""Render a Scorecard to JSON/markdown and apply pass/fail gates."""
from __future__ import annotations

from agent_graph.assurance.harness import Scorecard


def scorecard_to_dict(sc: Scorecard) -> dict:
    return sc.model_dump()


def scorecard_to_markdown(sc: Scorecard) -> str:
    lines = [f"# Assurance scorecard — {sc.sut_id} on {sc.dataset_id}", ""]
    for case in sc.results:
        lines.append(f"## case {case.case_id}")
        lines.append("| metric | value | passed | reason |")
        lines.append("|---|---|---|---|")
        for m in case.metrics:
            mark = "✅" if m.passed else "❌"
            lines.append(f"| {m.name} | {m.value:.2f} | {mark} | {m.reason} |")
        lines.append("")
    return "\n".join(lines)


def apply_gates(sc: Scorecard) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for case in sc.results:
        for m in case.metrics:
            if not m.passed:
                failures.append(f"{case.case_id}/{m.name}: {m.reason}")
    return (len(failures) == 0), failures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assurance/test_report.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_graph/assurance/report.py tests/assurance/test_report.py
git commit -m "feat(assurance): scorecard JSON/markdown rendering + pass/fail gates

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: CLI + fixture dataset + Makefile + end-to-end smoke

**Files:**
- Create: `src/agent_graph/assurance/cli.py`
- Create: `src/agent_graph/assurance/datasets/pathology_smoke.json`
- Create: `Makefile`
- Test: `tests/assurance/test_cli_smoke.py`

**Interfaces:**
- Consumes: `AssuranceHarness` (Task 4); `apply_gates`, `scorecard_to_dict`, `scorecard_to_markdown` (Task 5); `PathologyAdapter` (Task 3); `FaithfulnessMetric` (Task 2).
- Produces:
  - `load_dataset(path: str) -> list[dict]` — reads a JSON list of cases.
  - `run_cli(sut_name: str, dataset_path: str, sut=None, metrics=None) -> int` — runs the harness, prints the markdown scorecard, writes `results/<sut>_scorecard.json`, returns exit code (0 pass / 1 gate failure). `sut`/`metrics` are injectable for offline tests.
  - `main(argv=None)` — argparse: `run --sut {sga,pathology} --dataset <path>`.

- [ ] **Step 1: Write the failing test**

Create `tests/assurance/test_cli_smoke.py`:

```python
import json
from pathlib import Path

from agent_graph.assurance.sut import Response
from agent_graph.assurance.metrics.base import MetricResult
from agent_graph.assurance.cli import load_dataset, run_cli


class _StubPathologySUT:
    id = "pathology"
    def run(self, case):
        return Response(text="plain explanation", contexts=[case["report_text"]])


class _PassMetric:
    name = "faithfulness"
    def applies_to(self, sut_id): return True
    def score(self, case, response):
        return MetricResult(name="faithfulness", value=1.0, passed=True, reason="ok")


class _FailMetric:
    name = "patient_safety"
    def applies_to(self, sut_id): return True
    def score(self, case, response):
        return MetricResult(name="patient_safety", value=0.0, passed=False,
                            reason="dropped finding")


def test_load_dataset_reads_json(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"id": "c1", "report_text": "biopsy"}]))
    ds = load_dataset(str(p))
    assert ds == [{"id": "c1", "report_text": "biopsy"}]


def test_run_cli_exit0_on_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"id": "c1", "report_text": "biopsy"}]))
    code = run_cli("pathology", str(p), sut=_StubPathologySUT(), metrics=[_PassMetric()])
    assert code == 0
    assert (tmp_path / "results" / "pathology_scorecard.json").exists()


def test_run_cli_exit1_on_gate_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"id": "c1", "report_text": "biopsy"}]))
    code = run_cli("pathology", str(p), sut=_StubPathologySUT(), metrics=[_FailMetric()])
    assert code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assurance/test_cli_smoke.py -q`
Expected: FAIL (ModuleNotFoundError: `agent_graph.assurance.cli`).

- [ ] **Step 3: Write the module, fixture, and Makefile**

Create `src/agent_graph/assurance/datasets/pathology_smoke.json`:
```json
[
  {"id": "path-1", "report_text": "Skin, left forearm, punch biopsy: benign intradermal nevus. No atypia. Margins clear."},
  {"id": "path-2", "report_text": "Breast, core needle biopsy: invasive ductal carcinoma, grade 2. Lymphovascular invasion present."}
]
```

Create `src/agent_graph/assurance/cli.py`:
```python
"""CLI for the assurance harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_graph.assurance.harness import AssuranceHarness
from agent_graph.assurance.report import (
    apply_gates, scorecard_to_dict, scorecard_to_markdown)


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _build_sut(sut_name: str):
    if sut_name == "sga":
        from agent_graph.assurance.adapters.sga import SGAAdapter
        return SGAAdapter()
    if sut_name == "pathology":
        from agent_graph.assurance.adapters.pathology import PathologyAdapter
        return PathologyAdapter()
    raise ValueError(f"unknown sut {sut_name!r}")


def _default_metrics():
    from agent_graph.assurance.metrics.legacy import FaithfulnessMetric
    return [FaithfulnessMetric()]


def run_cli(sut_name: str, dataset_path: str, sut=None, metrics=None) -> int:
    sut = sut or _build_sut(sut_name)
    metrics = metrics if metrics is not None else _default_metrics()
    dataset = load_dataset(dataset_path)
    sc = AssuranceHarness().run(sut, dataset, metrics, dataset_id=Path(dataset_path).stem)

    print(scorecard_to_markdown(sc))
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sut.id}_scorecard.json").write_text(
        json.dumps(scorecard_to_dict(sc), indent=2))

    passed, failures = apply_gates(sc)
    if not passed:
        print("\nGATE FAILURES:")
        for line in failures:
            print(f"  - {line}")
    return 0 if passed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent_graph.assurance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--sut", choices=["sga", "pathology"], required=True)
    run_p.add_argument("--dataset", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run_cli(args.sut, args.dataset)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

Create `Makefile`:
```makefile
.PHONY: test smoke assure

test:
	python -m pytest tests/assurance -q

smoke:
	python -m pytest tests/assurance/test_cli_smoke.py -q

assure:
	python -m agent_graph.assurance run --sut pathology \
		--dataset src/agent_graph/assurance/datasets/pathology_smoke.json
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assurance/test_cli_smoke.py -q`
Expected: 3 passed.

- [ ] **Step 5: Run the full Phase-1 suite**

Run: `python -m pytest tests/assurance -q`
Expected: all pass (13 tests across the phase).

- [ ] **Step 6: Commit**

```bash
git add src/agent_graph/assurance/cli.py src/agent_graph/assurance/datasets/pathology_smoke.json \
        Makefile tests/assurance/test_cli_smoke.py
git commit -m "feat(assurance): CLI run command + fixture dataset + Makefile gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review (completed at plan-writing time)

- **Spec coverage (Phase 1 subset):** SystemUnderTest/Metric protocols (spec §2, §3) → Task 1; legacy metric wrap (§4) → Task 2; SGA + pathology adapters (§3) → Task 3; harness runner + Scorecard with per-adapter metric filtering (§2, §4) → Task 4; report + gates (§11) → Task 5; CLI + Makefile local gate + fixture (§11) → Task 6. Deferred to later phases (documented, not gaps): DeepEval + custom-clinical + RAG metrics (§4 Phase 2), PySpark ETL/store/drift (§7–8 Phase 3), Langfuse + DeepTeam red-team + docker/CI (§9–11 Phase 4), findings-first dataset generation (§5–6 Phase 3).
- **Placeholder scan:** none; every step has runnable code/commands.
- **Type consistency:** `Response`/`MetricResult` (Task 1) consumed unchanged in Tasks 2–6; `Metric` triple (`name`, `applies_to`, `score`) implemented identically by `FaithfulnessMetric` (Task 2) and satisfied by the stub metrics; `Scorecard`/`CaseResult` (Task 4) consumed by report (Task 5) and cli (Task 6); adapter `run(case)->Response` matches harness usage.

## Phasing (rest of the spec, each its own plan when reached)

- **Phase 2 — Metric expansion:** DeepEval backbone (G-Eval/Hallucination/Bias/Toxicity/Contextual\*, judge=`ChatAnthropic`), custom clinical metrics (ClinicalAppropriateness, PatientSafety, DiagnosisSycophancy), consistency@k wrap. Verify DeepEval's current custom-model + metric API at plan time.
- **Phase 3 — Data & monitoring:** findings-first pathology dataset + literature gold answers, PySpark ETL (lineage/versioning), run-store, PSI/KS drift. Verify PySpark local-mode API at plan time.
- **Phase 4 — Red-team, observability, infra:** DeepTeam attack suite, Langfuse exporter + docker-compose, optional Dockerfile, minimal GitHub workflow, README competency-map. Verify DeepTeam + Langfuse APIs at plan time.
