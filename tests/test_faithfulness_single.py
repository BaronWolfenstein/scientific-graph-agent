"""Single-call faithfulness judge (cheaper GEPA-metric variant): one LLM call
instead of 1 + N-claim verification, cutting per-eval cost ~3-4x."""
import pytest


class _FakeStructured:
    def __init__(self, result): self.result = result
    def invoke(self, _messages): return self.result


class _FakeLLM:
    def __init__(self, result): self.result = result; self.calls = 0
    def with_structured_output(self, _schema):
        self.calls += 1
        return _FakeStructured(self.result)


def test_single_call_returns_score_tuple_with_one_call():
    from agent_graph.eval.faithfulness import compute_faithfulness_single, FaithfulnessScore
    llm = _FakeLLM(FaithfulnessScore(supported_fraction=0.9, reasoning="ok"))
    score, n_sup, n_tot = compute_faithfulness_single(
        "summary text", [{"title": "T", "summary": "abstract"}], llm=llm)
    assert score == 0.9
    assert (n_sup, n_tot) == (0, 0)   # tuple shape matches compute_faithfulness
    assert llm.calls == 1              # exactly one LLM call


def test_single_call_guards_empty_inputs():
    from agent_graph.eval.faithfulness import compute_faithfulness_single
    assert compute_faithfulness_single("", [{"title": "T"}]) == (1.0, 0, 0)
    assert compute_faithfulness_single("s", []) == (1.0, 0, 0)
