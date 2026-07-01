"""Smoke test: the DSPy program + offline harness import and construct.

The real GEPA optimization run is offline/API-bound and is NOT exercised here.
"""
import pytest


def test_program_constructs():
    from agent_graph.optimize.program import DualAudienceProgram, GenerateDualAudience
    prog = DualAudienceProgram()
    assert prog is not None
    # Signature carries the grounding instruction GEPA will evolve from
    assert "PMIDs" in GenerateDualAudience.__doc__


def test_harness_exposes_compile_entrypoint():
    from agent_graph.optimize.run_gepa import compile_program, build_lm
    assert callable(compile_program) and callable(build_lm)
