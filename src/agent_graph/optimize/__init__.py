"""GEPA-based offline optimization of the summarizer prompt.

Decoupled from the knowledge graph: uses only the pre-HITL validation invariants
(schema + citation grounding) and the existing LLM-as-judge evaluators. Nothing
here is imported by the live LangGraph pipeline.
"""
