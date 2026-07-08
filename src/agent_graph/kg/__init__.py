# src/agent_graph/kg/__init__.py
"""Domain knowledge graph package."""
from agent_graph.kg.protocol import KnowledgeGraph
from agent_graph.kg.rdfstar_store import OxigraphKG


def get_knowledge_graph() -> KnowledgeGraph:
    """Factory — the single place that picks the backend."""
    return OxigraphKG()


def merge_graphs(a, b):
    """LangGraph reducer: fold parallel map-reduce branches into one graph."""
    if a is None:
        return b if b is not None else OxigraphKG()
    if b is None:
        return a
    a.merge_from(b)
    return a
