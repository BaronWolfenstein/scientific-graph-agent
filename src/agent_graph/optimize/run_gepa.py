"""OFFLINE GEPA compile step for the dual-audience summarizer prompt.

NOT imported by the live pipeline. Consumes an Anthropic budget and a training
set; run it periodically, review the evolved instruction, then paste it into
`dual_audience_node`.

Usage (from a script or notebook):

    from agent_graph.optimize.run_gepa import compile_program
    trainset = [dspy.Example(query=q, papers=p).with_inputs("query", "papers")
                for q, p in harvested_examples]
    optimized = compile_program(trainset, valset=heldout)
    print(optimized.clinician.signature.instructions)   # -> paste into CLINICIAN_GUIDANCE
    print(optimized.technical.signature.instructions)   # -> paste into TECHNICAL_GUIDANCE
"""
import os
import dspy

from agent_graph.optimize.program import DualAudienceProgram
from agent_graph.optimize.metric import summarizer_metric
from agent_graph.eval.faithfulness import compute_faithfulness_single

MODEL = "anthropic/claude-sonnet-4-6"


def build_lm(max_tokens: int = 8192):
    """Claude LM for both generation and GEPA's reflection step."""
    return dspy.LM(MODEL, api_key=os.environ["ANTHROPIC_API_KEY"], max_tokens=max_tokens)


def compile_program(trainset, valset=None, max_metric_calls: int = 30,
                    faithfulness_fn=compute_faithfulness_single):
    """Run GEPA to evolve the summarizer instruction. Returns the optimized module.

    `trainset`/`valset` are lists of dspy.Example(query=..., papers=...) with the
    gold also carrying `.papers` (list[dict] with pmids) for the metric's grounding
    and judge checks. Keep a held-out `valset` to confirm generalization.

    `max_metric_calls` bounds cost/time explicitly (GEPA `auto='light'` budgeted
    ~748 rollouts / hours here). Raise it for a more thorough search.

    `faithfulness_fn` defaults to the single-call variant (compute_faithfulness_single)
    to cut per-eval cost ~3-4x; pass compute_faithfulness for the multi-call judge.
    """
    lm = build_lm()
    dspy.configure(lm=lm)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return summarizer_metric(gold, pred, trace, pred_name, pred_trace,
                                 faithfulness_fn=faithfulness_fn)

    optimizer = dspy.GEPA(metric=metric, reflection_lm=lm,
                          max_metric_calls=max_metric_calls)
    return optimizer.compile(DualAudienceProgram(), trainset=trainset,
                             valset=valset or trainset)


if __name__ == "__main__":
    raise SystemExit(
        "run_gepa is an offline library step. Import compile_program(trainset) with "
        "a harvested trainset of dspy.Example(query=..., papers=...). It consumes "
        "API budget and is not run by tests or the live pipeline."
    )
