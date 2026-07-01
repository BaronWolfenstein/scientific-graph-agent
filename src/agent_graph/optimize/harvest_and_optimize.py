"""OFFLINE, READY-TO-RUN: harvest (query, papers) from demo runs, then GEPA-optimize
the dual-audience summarizer prompt.

This is NOT run by tests or the live pipeline. It consumes ANTHROPIC_API_KEY budget
and hits PubMed. The demo checkpointer is in-memory (runs are not persisted), so we
regenerate examples by running the demo graph on seed queries up to the HITL
interrupt and reading `papers` from the graph state — no approval needed for harvest.

Run:
    python -m agent_graph.optimize.harvest_and_optimize

Then review the printed evolved instruction and paste it into
`dual_audience_node`'s system prompt (or load it as a prompt asset).
"""
import os
import dspy
from pathlib import Path
from dotenv import load_dotenv

from agent_graph.graph import create_demo_graph
from agent_graph.optimize.run_gepa import compile_program

# pick up ANTHROPIC_API_KEY from the repo-root .env (src/agent_graph/optimize/ -> repo root)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Deliberately cross-family so GEPA evolves an instruction that generalizes
# (oncology, cardiology, endocrine/metabolic, neurology, and PATHOLOGY). Scale
# this up (~20-40, several per family) before trusting a single prompt; the
# held-out valset measures per-family generalization (see spec 2026-06-30 §6.1).
SEED_QUERIES = [
    # oncology (therapy)
    "CAR-T cell therapy efficacy and safety in relapsed refractory diffuse large B-cell lymphoma",
    "efficacy of pembrolizumab in triple-negative breast cancer",
    # cardiology / metabolic
    "SGLT2 inhibitors and cardiovascular outcomes in heart failure",
    "GLP-1 receptor agonists for weight loss in non-diabetic adults",
    "statins for primary prevention of cardiovascular disease in older adults",
    # neurology
    "anti-amyloid monoclonal antibodies for early Alzheimer's disease",
    # pathology (diagnostic / computational / molecular)
    "diagnostic accuracy of PD-L1 immunohistochemistry for predicting checkpoint-inhibitor response in NSCLC",
    "deep learning for Gleason grading of prostate biopsies in digital pathology",
    "methylation-profiling molecular classification of diffuse gliomas",
]


def harvest(queries=SEED_QUERIES, max_papers=4):
    """Run the demo graph to the HITL interrupt on each query; collect
    dspy.Example(query, papers). `papers` stays a list[dict] so the metric can read
    PMIDs; the program formats it to text at generation time."""
    examples = []
    for i, q in enumerate(queries):
        try:
            graph = create_demo_graph()
            config = {"configurable": {"thread_id": f"harvest-{i}"}}
            graph.invoke({"query": q, "max_papers": max_papers}, config=config)
            papers = graph.get_state(config).values.get("papers", [])
        except Exception as exc:  # one bad query must not abort the whole harvest
            print(f"  SKIP '{q[:50]}...': {type(exc).__name__}: {exc}")
            continue
        if papers:
            examples.append(
                dspy.Example(query=q, papers=papers).with_inputs("query", "papers")
            )
        print(f"  harvested {len(papers)} papers for: {q[:60]}...")
    return examples


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY first (or load a .env).")

    print("Harvesting examples from demo runs (uses PubMed + API budget)...")
    examples = harvest()
    if len(examples) < 2:
        raise SystemExit(f"Harvested only {len(examples)} usable examples; need >= 2.")

    split = max(1, int(len(examples) * 0.7))
    trainset = examples[:split]
    valset = examples[split:] or examples[:1]
    print(f"\n{len(examples)} examples ({len(trainset)} train / {len(valset)} val). "
          f"Running GEPA (auto='light')...\n")

    optimized = compile_program(trainset, valset=valset, auto="light")

    print("\n=== EVOLVED CLINICIAN INSTRUCTION (paste into CLINICIAN_GUIDANCE) ===\n")
    print(optimized.clinician.signature.instructions)
    print("\n=== EVOLVED TECHNICAL INSTRUCTION (paste into TECHNICAL_GUIDANCE) ===\n")
    print(optimized.technical.signature.instructions)


if __name__ == "__main__":
    main()
