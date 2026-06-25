"""
CLI: measure Consistency@k for a query.

Usage:
    python eval_consistency.py "transformer attention mechanisms" --k 5
    python eval_consistency.py "CRISPR gene editing" --k 3 --max-papers 5 --model gpt-4o-mini
"""
import argparse
import logging
logging.basicConfig(level=logging.WARNING)

import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).parent
os.chdir(_project_root)
load_dotenv(_project_root / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from agent_graph.eval.consistency import run_consistency_eval


def main():
    parser = argparse.ArgumentParser(description="Consistency@k evaluation")
    parser.add_argument("query", help="Query to evaluate")
    parser.add_argument("--k", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--max-papers", type=int, default=5, help="Papers per run (default: 5)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--tools", default="arxiv", help="Comma-separated tools (default: arxiv)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--relevance-threshold", type=int, default=50, help="Min relevance_score to count as relevant for Context Precision (default: 50)")
    args = parser.parse_args()

    tools = [t.strip() for t in args.tools.split(",")]

    print(f"\nRunning Consistency@{args.k} evaluation...")
    print(f"Query: {args.query}\n")

    report = run_consistency_eval(
        query=args.query,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        graph_kwargs={"tools": tools},
        invoke_kwargs={"max_papers": args.max_papers, "llm_model": args.model, "llm_temperature": args.temperature},
    )

    print(report.summary())


if __name__ == "__main__":
    main()
