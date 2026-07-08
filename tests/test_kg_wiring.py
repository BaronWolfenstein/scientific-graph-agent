"""Wiring tests: state field + summarizer context injection + demo graph shape."""
import pytest


def test_state_has_knowledge_graph_field():
    from agent_graph.state import InternalState
    assert "knowledge_graph" in InternalState.__annotations__


def test_demo_graph_includes_graph_builder():
    from agent_graph.graph import create_demo_graph
    g = create_demo_graph()
    assert "graph_builder" in g.get_graph().nodes


def test_summarizer_injects_graph_context(monkeypatch):
    import agent_graph.nodes as nodes
    from agent_graph.kg import get_knowledge_graph

    captured = {}

    class _LLM:
        def invoke(self, messages):
            captured["messages"] = messages
            from langchain_core.messages import AIMessage
            return AIMessage(content="## Summary\n• ok [Paper 1]")

    monkeypatch.setattr(nodes, "get_llm", lambda **kw: _LLM())

    kg = get_knowledge_graph()
    kg.add_relation("Imatinib", "drug", "treats", "CML", "disease",
                    {"paper_uri": "pmid:1", "pmid": "1", "paper_id": "1",
                     "pub_year": 2020, "relevance": 90, "polarity": "supports", "snippet": "t"})
    state = {
        "papers": [{"id": "1", "pmid": "1", "title": "T", "authors": ["A"],
                    "published": "2020", "summary": "s", "url": "u", "relevance_score": 90}] * 3,
        "query": "imatinib cml", "iteration": 0, "max_iterations": 2,
        "knowledge_graph": kg,
    }
    nodes.summarizer_node(state)
    blob = "\n".join(str(getattr(m, "content", "")) for m in captured["messages"])
    assert "literature graph" in blob and "imatinib" in blob
