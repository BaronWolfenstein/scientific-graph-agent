"""Answer Relevance evaluation: measures whether the summary addresses the user's query."""
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from agent_graph.llm import get_llm


class RelevanceScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="How well the summary answers the query (1=not at all, 5=completely)")
    reasoning: str = Field(..., description="One sentence explaining the score")


def compute_answer_relevance(query: str, summary: str) -> tuple[float, str]:
    """
    Score how well the summary answers the original user query.

    Uses a single LLM call with structured output.

    Args:
        query: The original user query.
        summary: The generated summary.

    Returns:
        (score_0_to_1, reasoning) where score is (raw_score - 1) / 4 normalised to [0, 1].
    """
    if not query or not summary:
        return 0.0, "missing query or summary"

    llm = get_llm(temperature=0)
    structured = llm.with_structured_output(RelevanceScore)
    result = structured.invoke([
        SystemMessage(content=(
            "You are evaluating a scientific research assistant. "
            "Score how well the provided summary answers the user's original query. "
            "1 = the summary is completely off-topic or doesn't address the query at all. "
            "2 = the summary is tangentially related but misses the core question. "
            "3 = the summary partially answers the query but has significant gaps. "
            "4 = the summary mostly answers the query with minor gaps. "
            "5 = the summary directly and completely addresses the query."
        )),
        HumanMessage(content=f"User query: {query}\n\nSummary:\n{summary}"),
    ])
    normalised = (result.score - 1) / 4
    return normalised, result.reasoning
