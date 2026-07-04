"""The dual-audience guidance is a named, swappable prompt asset.

GEPA's evolved clinician/technical instructions deploy by editing
CLINICIAN_GUIDANCE / TECHNICAL_GUIDANCE — the system prompt is composed from the
guidance + grounding rule + JSON schema (which stays fixed, as the gate contract).
"""
import pytest


def test_audience_system_prompt_composes_guidance_grounding_schema():
    from agent_graph.nodes import (
        _audience_system_prompt, CLINICIAN_GUIDANCE, TECHNICAL_GUIDANCE,
    )
    prompt = _audience_system_prompt(CLINICIAN_GUIDANCE, "GROUNDING_RULE", "SCHEMA_STR")
    assert CLINICIAN_GUIDANCE in prompt
    assert "GROUNDING_RULE" in prompt
    assert "SCHEMA_STR" in prompt
    assert isinstance(TECHNICAL_GUIDANCE, str) and TECHNICAL_GUIDANCE
