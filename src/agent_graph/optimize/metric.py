"""GEPA metric for the dual-audience summarizer.

Combines hard correctness gates (JSON Schema + citation grounding — the same
invariants as the pre-HITL gate) with LLM-as-judge quality (faithfulness +
answer-relevance, reused from `agent_graph.eval`). Returns a
`dspy.Prediction(score, feedback)` so GEPA can reflect on the textual diagnostic.

The two judges are injected (defaulting to the real evaluators) so the
deterministic gate logic is unit-testable without any API calls.

Scoring:
  - schema invalid            -> score 0.0 (structure must be fixed first)
  - else quality * grounding, where
      quality  = 0.5*faithfulness + 0.5*answer_relevance   (answer_relevance
                 defeats the degenerate "say nothing / stay faithful" summary)
      grounding = 1.0 if all cited PMIDs were retrieved else 0.3
"""
import json
import jsonschema
import dspy

from agent_graph.schemas import ClinicianSummary, TechnicalSummary
from agent_graph.eval.faithfulness import compute_faithfulness
from agent_graph.eval.answer_relevance import compute_answer_relevance

_CLINICIAN_SCHEMA = ClinicianSummary.model_json_schema()
_TECHNICAL_SCHEMA = TechnicalSummary.model_json_schema()


def _schema_errors(cs, ts):
    errors = []
    for name, obj, schema in [("clinician_summary", cs, _CLINICIAN_SCHEMA),
                              ("technical_summary", ts, _TECHNICAL_SCHEMA)]:
        if obj is None:
            errors.append(f"{name}: missing")
            continue
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{name}: {exc.message}")
    return errors


def _ungrounded(cs, ts, papers):
    allowed = {p.get("pmid") for p in papers if p.get("pmid")}
    bad = []
    if allowed:
        for obj in (cs, ts):
            for ev in (obj or {}).get("evidence", []):
                pmid = ev.get("pmid")
                if pmid and pmid not in allowed:
                    bad.append(pmid)
    return bad


def _as_dict(x):
    """DSPy typed outputs may arrive as JSON strings; coerce to dict or None."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (ValueError, TypeError):
            return None
    return x


def _render_text(cs, ts):
    parts = []
    if cs:
        parts.append(cs.get("bottom_line", ""))
        parts.extend(cs.get("key_findings", []))
        parts.append(cs.get("confidence_note", ""))
    if ts:
        parts.append(ts.get("detailed_findings", ""))
        parts.append(ts.get("methodology_notes", ""))
        parts.extend(ts.get("caveats", []))
    return "\n".join(p for p in parts if p)


def summarizer_metric(gold, pred, trace=None, pred_name=None, pred_trace=None,
                      *, faithfulness_fn=compute_faithfulness,
                         relevance_fn=compute_answer_relevance):
    """GEPA metric: dspy.Prediction(score in [0,1], feedback str)."""
    cs = _as_dict(getattr(pred, "clinician_summary", None))
    ts = _as_dict(getattr(pred, "technical_summary", None))
    papers = getattr(gold, "papers", None) or []
    query = getattr(gold, "query", "")

    schema_errors = _schema_errors(cs, ts)
    if schema_errors:
        return dspy.Prediction(
            score=0.0,
            feedback="Schema invalid — fix structure first: " + "; ".join(schema_errors),
        )

    ungrounded = _ungrounded(cs, ts, papers)
    text = _render_text(cs, ts)
    faith, _n_sup, _n_tot = faithfulness_fn(text, papers)
    rel, rel_reason = relevance_fn(query, text)

    quality = 0.5 * faith + 0.5 * rel
    grounding_factor = 1.0 if not ungrounded else 0.3
    score = quality * grounding_factor

    fb = [f"faithfulness={faith:.2f}", f"answer_relevance={rel:.2f}: {rel_reason}"]
    if ungrounded:
        fb.insert(0, f"UNGROUNDED citations not in retrieved papers: "
                     f"{sorted(set(ungrounded))} — cite only retrieved PMIDs")
    return dspy.Prediction(score=score, feedback=" | ".join(fb))
