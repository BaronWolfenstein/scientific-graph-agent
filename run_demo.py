"""
Demo entry point: one query end-to-end with HITL approval.

Usage:
    python3 run_demo.py
    python3 run_demo.py "efficacy of pembrolizumab in triple-negative breast cancer"

The script runs until the HITL interrupt, shows both draft summaries,
prompts for approve/reject, then finalizes or discards.
"""
import sys
import logging
logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during demo

from langgraph.types import Command
from agent_graph.graph import create_demo_graph

DEMO_QUERY = (
    "CAR-T cell therapy efficacy and safety in relapsed refractory "
    "diffuse large B-cell lymphoma"
)


def _print_section(title: str, content: str) -> None:
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    print(content)


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEMO_QUERY
    print(f"\nScientific Graph Agent — PubMed + Anthropic Demo")
    print(f"Query: {query}\n")

    graph = create_demo_graph()
    config = {"configurable": {"thread_id": "demo-session-1"}}

    print("Running pipeline (clarifier → PubMed → summarizer → dual-audience)...")
    result = graph.invoke({"query": query, "max_papers": 4}, config=config)

    # Check whether we stopped at the HITL interrupt
    snapshot = graph.get_state(config)
    if snapshot.next:
        # Extract interrupt payload from pending tasks
        interrupt_payload = {}
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break

        display = interrupt_payload.get("display", "")
        message = interrupt_payload.get("message", "")

        _print_section("DRAFT SUMMARIES FOR REVIEW", display)
        print(message)

        while True:
            decision = input("Decision (approve / reject): ").strip().lower()
            if decision in ("approve", "reject"):
                break
            print("Please type 'approve' or 'reject'.")

        print(f"\nResuming with: {decision}...")
        result = graph.invoke(
            Command(resume={"action": decision}),
            config=config
        )

    # Print final state
    if result.get("approved") is False or (result.get("summary") or "").startswith("[REJECTED"):
        print("\nSummaries rejected — pipeline ended without output.")
        return

    cs = result.get("clinician_summary")
    ts = result.get("technical_summary")

    if cs:
        _print_section("CLINICIAN SUMMARY (FINAL)", "")
        print(f"Bottom line : {cs.get('bottom_line', '')}\n")
        print("Key findings:")
        for f in cs.get("key_findings", []):
            print(f"  • {f}")
        print(f"\nConfidence  : {cs.get('confidence_note', '')}")
        print("\nEvidence:")
        for e in cs.get("evidence", []):
            print(f"  [{e.get('pmid','')}] {e.get('source_url','')}")

    if ts:
        _print_section("TECHNICAL SUMMARY (FINAL)", "")
        print(ts.get("detailed_findings", ""))
        print(f"\nMethodology : {ts.get('methodology_notes', '')}")
        print("\nCaveats:")
        for c in ts.get("caveats", []):
            print(f"  • {c}")
        print("\nEvidence:")
        for e in ts.get("evidence", []):
            print(f"  [{e.get('pmid','')}] {e.get('source_url','')}")


if __name__ == "__main__":
    main()
