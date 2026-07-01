"""OFFLINE GEPA compile step for the dual-audience summarizer prompt.

NOT imported by the live pipeline. Consumes an Anthropic budget and a training
set; run it periodically, review the evolved instruction, then paste it into
`dual_audience_node`.

Usage (from a script or notebook):

    from agent_graph.optimize.run_gepa import compile_program
    trainset = [dspy.Example(query=q, papers=p).with_inputs("query", "papers")
                for q, p in harvested_examples]
    optimized = compile_program(trainset, valset=heldout)
    print(optimized.generate.signature.instructions)   # the evolved prompt
"""
import os
import dspy

from agent_graph.optimize.program import DualAudienceProgram
from agent_graph.optimize.metric import summarizer_metric

MODEL = "anthropic/claude-sonnet-4-6"


def build_lm(max_tokens: int = 8192):
    """Claude LM for both generation and GEPA's reflection step."""
    return dspy.LM(MODEL, api_key=os.environ["ANTHROPIC_API_KEY"], max_tokens=max_tokens)


def compile_program(trainset, valset=None, auto: str = "light"):
    """Run GEPA to evolve the summarizer instruction. Returns the optimized module.

    `trainset`/`valset` are lists of dspy.Example(query=..., papers=...) with the
    gold also carrying `.papers` (list[dict] with pmids) for the metric's grounding
    and judge checks. Keep a held-out `valset` to confirm generalization.
    """
    lm = build_lm()
    dspy.configure(lm=lm)
    optimizer = dspy.GEPA(metric=summarizer_metric, reflection_lm=lm, auto=auto)
    return optimizer.compile(DualAudienceProgram(), trainset=trainset,
                             valset=valset or trainset)


if __name__ == "__main__":
    raise SystemExit(
        "run_gepa is an offline library step. Import compile_program(trainset) with "
        "a harvested trainset of dspy.Example(query=..., papers=...). It consumes "
        "API budget and is not run by tests or the live pipeline."
    )
