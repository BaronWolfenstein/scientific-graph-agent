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


def test_format_papers_renders_pmid_and_title():
    from agent_graph.optimize.program import DualAudienceProgram
    text = DualAudienceProgram._format_papers(
        [{"pmid": "111", "title": "Seurat v3", "summary": "abstract", "url": "u"}]
    )
    assert "111" in text and "Seurat v3" in text
    # a pre-formatted string passes through unchanged
    assert DualAudienceProgram._format_papers("already text") == "already text"


def test_harvest_script_imports():
    # offline script must import cleanly (does not run — that needs API + PubMed)
    from agent_graph.optimize.harvest_and_optimize import harvest, main, SEED_QUERIES
    assert callable(harvest) and callable(main) and len(SEED_QUERIES) >= 2
