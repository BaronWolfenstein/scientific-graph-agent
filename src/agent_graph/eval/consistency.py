"""Consistency@k evaluation: measures reproducibility across independent runs."""
from dataclasses import dataclass
from typing import Optional
import statistics
import time

from langchain_core.globals import set_llm_cache
from agent_graph.graph import create_graph
from agent_graph.eval.faithfulness import compute_faithfulness
from agent_graph.eval.answer_relevance import compute_answer_relevance


@dataclass
class RunResult:
    paper_ids: set
    paper_titles: list
    paper_scores: list          # relevance_score per paper, in ranked order
    refined_query: Optional[str]
    paper_count: int
    context_precision: float    # average precision over ranked list at relevance_threshold
    mrr: float                  # reciprocal rank of first relevant result
    faithfulness: float         # fraction of summary claims supported by retrieved papers
    faithfulness_detail: tuple  # (score, n_supported, n_total)
    answer_relevance: float     # 0-1, how well the summary addresses the original query
    answer_relevance_reason: str
    summary: Optional[str] = None


@dataclass
class ConsistencyReport:
    query: str
    k: int
    runs: list
    mean_jaccard: float
    min_jaccard: float
    query_unique_count: int
    paper_union_size: int
    paper_intersection_size: int
    mean_context_precision: float
    min_context_precision: float
    mean_faithfulness: float
    min_faithfulness: float
    mean_answer_relevance: float
    min_answer_relevance: float
    mean_mrr: float
    min_mrr: float

    def summary(self) -> str:
        lines = [
            f"Query              : {self.query}",
            f"Runs (k)           : {self.k}",
            f"Mean Jaccard       : {self.mean_jaccard:.3f}  (min: {self.min_jaccard:.3f})",
            f"Mean Ctx Precision : {self.mean_context_precision:.3f}  (min: {self.min_context_precision:.3f})",
            f"Mean Faithfulness  : {self.mean_faithfulness:.3f}  (min: {self.min_faithfulness:.3f})",
            f"Mean Ans Relevance : {self.mean_answer_relevance:.3f}  (min: {self.min_answer_relevance:.3f})",
            f"Mean MRR           : {self.mean_mrr:.3f}  (min: {self.min_mrr:.3f})",
            f"Paper union        : {self.paper_union_size}  intersection: {self.paper_intersection_size}",
            f"Unique refined queries: {self.query_unique_count} / {self.k}",
            "",
            "Per-run details:",
        ]
        for i, run in enumerate(self.runs):
            q = run.refined_query or "(not captured)"
            lines.append(
                f"  Run {i+1}: {run.paper_count} papers | "
                f"ctx_precision={run.context_precision:.3f} | mrr={run.mrr:.3f} | query: {q!r}"
            )
            for title, score in zip(run.paper_titles, run.paper_scores):
                lines.append(f"    [{score:3d}] {title}")
            _, n_sup, n_tot = run.faithfulness_detail
            lines.append(f"    faithfulness: {run.faithfulness:.3f} ({n_sup}/{n_tot} claims supported)")
            lines.append(f"    ans_relevance: {run.answer_relevance:.3f} — {run.answer_relevance_reason}")
            if run.summary:
                lines.append(f"  Summary: {run.summary[:200]}...")
        return "\n".join(lines)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mrr(scores: list, threshold: int = 50) -> float:
    """Reciprocal rank of the first relevant result (score >= threshold). 0.0 if none found."""
    for rank, score in enumerate(scores, start=1):
        if score >= threshold:
            return 1.0 / rank
    return 0.0


def _context_precision(scores: list, threshold: int = 50) -> float:
    """
    Average precision over a ranked list using relevance_score as the relevance signal.

    A paper is considered relevant if its relevance_score >= threshold (1-100 scale).
    Computes precision at each rank k where the paper is relevant, then averages.
    Returns 1.0 for an empty list (no retrieval attempted).
    """
    if not scores:
        return 1.0

    relevant_seen = 0
    precision_at_relevant = []
    for rank, score in enumerate(scores, start=1):
        if score >= threshold:
            relevant_seen += 1
            precision_at_relevant.append(relevant_seen / rank)

    if not precision_at_relevant:
        return 0.0
    return sum(precision_at_relevant) / len(precision_at_relevant)


def run_consistency_eval(
    query: str,
    k: int = 5,
    relevance_threshold: int = 50,
    graph_kwargs: Optional[dict] = None,
    invoke_kwargs: Optional[dict] = None,
) -> ConsistencyReport:
    """
    Run the agent graph k times on the same query and measure result consistency.

    Args:
        query: User query to evaluate.
        k: Number of independent runs.
        relevance_threshold: Minimum relevance_score (1-100) to count a paper as relevant
                             when computing Context Precision. Default 50.
        graph_kwargs: Forwarded to create_graph (e.g. tools=["arxiv"], mode="sequential").
        invoke_kwargs: Extra initial-state fields (e.g. max_papers=5, llm_model="gpt-4o-mini").

    Returns:
        ConsistencyReport with pairwise Jaccard similarity, Context Precision, and per-run details.
    """
    graph_kwargs = graph_kwargs or {}
    invoke_kwargs = invoke_kwargs or {}

    # Disable the module-level SQLiteCache so repeated runs aren't served from cache
    set_llm_cache(None)

    runs: list[RunResult] = []
    for i in range(k):
        print(f"Run {i+1}/{k}...")
        graph = create_graph(with_checkpointer=False, **graph_kwargs)
        result = graph.invoke({"query": query, **invoke_kwargs})
        if i < k - 1:
            time.sleep(3)

        papers = result.get("papers", [])
        # papers are already sorted by relevance_score descending (keep_top_papers)
        paper_ids = {p["id"] for p in papers}
        scores = [p.get("relevance_score", 0) for p in papers]
        run_summary = result.get("summary")
        faith_score, faith_sup, faith_tot = compute_faithfulness(run_summary or "", papers)
        ans_rel_score, ans_rel_reason = compute_answer_relevance(query, run_summary or "")
        runs.append(RunResult(
            paper_ids=paper_ids,
            paper_titles=[p.get("title", p["id"]) for p in papers],
            paper_scores=scores,
            refined_query=result.get("refined_query"),
            paper_count=len(paper_ids),
            context_precision=_context_precision(scores, relevance_threshold),
            mrr=_mrr(scores, relevance_threshold),
            faithfulness=faith_score,
            faithfulness_detail=(faith_score, faith_sup, faith_tot),
            answer_relevance=ans_rel_score,
            answer_relevance_reason=ans_rel_reason,
            summary=run_summary,
        ))

    # Pairwise Jaccard across all (i, j) pairs with i < j
    pairwise = [
        _jaccard(runs[i].paper_ids, runs[j].paper_ids)
        for i in range(k)
        for j in range(i + 1, k)
    ]
    mean_j = statistics.mean(pairwise) if pairwise else 1.0
    min_j = min(pairwise) if pairwise else 1.0

    all_sets = [r.paper_ids for r in runs]
    union = set().union(*all_sets)
    intersection = all_sets[0].copy()
    for s in all_sets[1:]:
        intersection &= s

    cp_scores = [r.context_precision for r in runs]
    faith_scores = [r.faithfulness for r in runs]
    ar_scores = [r.answer_relevance for r in runs]
    mrr_scores = [r.mrr for r in runs]

    return ConsistencyReport(
        query=query,
        k=k,
        runs=runs,
        mean_jaccard=mean_j,
        min_jaccard=min_j,
        query_unique_count=len({r.refined_query for r in runs if r.refined_query is not None}),
        paper_union_size=len(union),
        paper_intersection_size=len(intersection),
        mean_context_precision=statistics.mean(cp_scores),
        min_context_precision=min(cp_scores),
        mean_faithfulness=statistics.mean(faith_scores),
        min_faithfulness=min(faith_scores),
        mean_answer_relevance=statistics.mean(ar_scores),
        min_answer_relevance=min(ar_scores),
        mean_mrr=statistics.mean(mrr_scores),
        min_mrr=min(mrr_scores),
    )
