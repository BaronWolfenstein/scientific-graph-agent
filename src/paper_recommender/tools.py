"""Tools for fetching recent papers."""
import arxiv
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor
import threading

_executor = None
_executor_lock = threading.Lock()


def _get_executor():
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=4)
    return _executor


def _fetch_recent_papers(topic: str, max_results: int) -> list[dict]:
    """Helper function to fetch recent papers from ArXiv."""
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=1.0,
        num_retries=1
    )

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
        })

    return papers


@tool
def search_recent_papers(topic: str, max_results: int = 5) -> list[dict]:
    """
    Search for recent papers on ArXiv by topic.

    Args:
        topic: Research topic (e.g., "transformers", "quantum computing")
        max_results: Maximum number of papers to return (default: 5)

    Returns:
        List of recent papers with metadata
    """
    executor = _get_executor()
    future = executor.submit(_fetch_recent_papers, topic, max_results)
    return future.result()
