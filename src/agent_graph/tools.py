"""ArXiv search and Wikipedia research tools with streaming support."""
import arxiv
import wikipedia
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Create a thread pool executor that will be reused
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

def _fetch_arxiv_results(query: str, max_results: int) -> list[dict]:
    """
    Helper function to fetch ArXiv results synchronously.
    This is run in a separate thread to avoid blocking the event loop.
    """
    # Create a custom client with more conservative rate limiting
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=1.0,  # Increased from default 3.0
        num_retries=1      # Increased retries
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
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


async def _fetch_arxiv_results_async(query: str, max_results: int, writer=None) -> list[dict]:
    """
    Async helper function to fetch ArXiv results with progress updates.

    Args:
        query: Search keywords
        max_results: Maximum number of papers to return
        writer: Stream writer for custom progress updates (optional)

    Returns:
        List of papers with structured metadata
    """

    papers = []
    if writer and callable(writer):
        result = writer(f"Starting ArXiv search for: '{query}'")

    loop = asyncio.get_running_loop()

    def _search_with_rate_limit():
        """Helper to run search with rate limiting."""

        # Create a custom client with more conservative rate limiting
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=1.0,  # Increased from default 3.0
            num_retries=1       # Increased retries
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        return list(client.results(search))

    results = await loop.run_in_executor(_get_executor(), _search_with_rate_limit)

    for i, result in enumerate(results, 1):
        paper = {
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
        }
        papers.append(paper)

        if writer and callable(writer):
            # Emit progress update
            writer(f"Found paper {i}/{len(results)}: {result.title[:60]}...")

    if writer and callable(writer):
        result = writer(f"✅ Search complete: {len(papers)} papers found")

    return papers


@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for scientific papers on ArXiv.

    This tool queries the ArXiv API and returns relevant papers
    with their metadata (title, authors, abstract, URL).

    Args:
        query: Search keywords (e.g., "transformer attention mechanism")
        max_results: Maximum number of papers to return (default: 5)

    Returns:
        List of papers with structured metadata
    """
    # Run the blocking operation in a thread pool to avoid blocking the event loop
    executor = _get_executor()
    future = executor.submit(_fetch_arxiv_results, query, max_results)
    return future.result()


@tool
async def search_arxiv_streaming(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for scientific papers on ArXiv with streaming progress updates.

    This async tool queries the ArXiv API and emits progress updates
    as papers are found. Use this within LangGraph to get real-time
    search progress.

    Args:
        query: Search keywords (e.g., "transformer attention mechanism")
        max_results: Maximum number of papers to return (default: 5)

    Returns:
        List of papers with structured metadata
    """

    try:
        writer = get_stream_writer()
        # Check if writer is actually callable
        if writer is None or not callable(writer):
            writer = None
    except Exception:
        # If not in LangGraph context, fall back to no streaming
        writer = None

    return await _fetch_arxiv_results_async(query, max_results, writer)


def _fetch_wikipedia_summary(topic: str, sentences: int = 5) -> dict:
    """
    Helper function to fetch Wikipedia summary synchronously.
    This is run in a separate thread to avoid blocking the event loop.
    """
    try:
        # Search for the topic
        search_results = wikipedia.search(topic, results=5)

        if not search_results:
            return {
                "success": False,
                "error": f"No Wikipedia articles found for '{topic}'",
                "topic": topic
            }

        # Get the first result
        page_title = search_results[0]

        # Fetch the page summary
        summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)
        page = wikipedia.page(page_title, auto_suggest=False)

        return {
            "success": True,
            "topic": topic,
            "title": page.title,
            "summary": summary,
            "url": page.url,
            "related_topics": search_results[1:5] if len(search_results) > 1 else []
        }

    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        return {
            "success": False,
            "error": "Multiple articles found. Please be more specific.",
            "topic": topic,
            "disambiguation_options": e.options[:10]  # Return first 10 options
        }

    except wikipedia.exceptions.PageError:
        return {
            "success": False,
            "error": f"Page not found for '{topic}'",
            "topic": topic
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching Wikipedia data: {str(e)}",
            "topic": topic
        }


async def _fetch_wikipedia_summary_async(topic: str, sentences: int = 5, writer=None) -> dict:
    """
    Async helper function to fetch Wikipedia summary with progress updates.

    Args:
        topic: Topic to search for
        sentences: Number of sentences in the summary (default: 5)
        writer: Stream writer for custom progress updates (optional)

    Returns:
        Dictionary with Wikipedia article data
    """
    if writer and callable(writer):
        writer(f"Searching Wikipedia for: '{topic}'")

    loop = asyncio.get_running_loop()

    try:
        # Run the synchronous Wikipedia fetch in a thread pool
        result = await loop.run_in_executor(_get_executor(), _fetch_wikipedia_summary, topic, sentences)

        if writer and callable(writer):
            if result["success"]:
                writer(f"✅ Found article: {result['title']}")
            else:
                if "disambiguation_options" in result:
                    writer(f"⚠️ Multiple articles found. Returning disambiguation options.")
                else:
                    writer(f"❌ {result['error']}")

        return result

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "topic": topic
        }
        if writer and callable(writer):
            writer(f"❌ Error: {str(e)}")
        return error_result


@tool
def search_wikipedia(topic: str, sentences: int = 5) -> dict:
    """
    Search Wikipedia for information about a topic.

    This tool searches Wikipedia and returns a summary of the most relevant article.
    If multiple articles match, it returns disambiguation options.

    Args:
        topic: The topic to search for (e.g., "quantum computing", "Albert Einstein")
        sentences: Number of sentences in the summary (default: 5)

    Returns:
        Dictionary containing:
        - success: Whether the search was successful
        - title: Article title (if found)
        - summary: Article summary (if found)
        - url: Wikipedia URL (if found)
        - related_topics: List of related article titles
        - error: Error message (if unsuccessful)
        - disambiguation_options: List of options if topic is ambiguous
    """
    executor = _get_executor()
    future = executor.submit(_fetch_wikipedia_summary, topic, sentences)
    return future.result()


@tool
async def search_wikipedia_streaming(topic: str, sentences: int = 5) -> dict:
    """
    Search Wikipedia for information about a topic with streaming progress updates.

    This async tool searches Wikipedia and emits progress updates during the search.
    Use this within LangGraph to get real-time search progress.

    Args:
        topic: The topic to search for (e.g., "quantum computing", "Albert Einstein")
        sentences: Number of sentences in the summary (default: 5)

    Returns:
        Dictionary containing:
        - success: Whether the search was successful
        - title: Article title (if found)
        - summary: Article summary (if found)
        - url: Wikipedia URL (if found)
        - related_topics: List of related article titles
        - error: Error message (if unsuccessful)
        - disambiguation_options: List of options if topic is ambiguous
    """
    try:
        writer = get_stream_writer()
        # Check if writer is actually callable
        if writer is None or not callable(writer):
            writer = None
    except Exception:
        # If not in LangGraph context, fall back to no streaming
        writer = None

    return await _fetch_wikipedia_summary_async(topic, sentences, writer)


def _fetch_pubmed_results(query: str, max_results: int) -> list[dict]:
    """Fetch PubMed results via NCBI E-utilities (esearch + efetch). No extra deps."""
    # Step 1: esearch — get PMIDs
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    })
    esearch_url = f"{NCBI_BASE}/esearch.fcgi?{params}"
    with urllib.request.urlopen(esearch_url, timeout=15) as r:
        data = json.loads(r.read())
    pmids = data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    # Step 2: efetch — get full records as XML
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })
    efetch_url = f"{NCBI_BASE}/efetch.fcgi?{params}"
    with urllib.request.urlopen(efetch_url, timeout=15) as r:
        xml_data = r.read()

    root = ET.fromstring(xml_data)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        year = article.findtext(".//PubDate/Year", "N/A")

        author_els = article.findall(".//Author")
        authors = []
        for a in author_els[:5]:
            ln = a.findtext("LastName", "")
            fn = a.findtext("ForeName", "")
            if ln:
                authors.append(f"{fn} {ln}".strip() if fn else ln)

        pmid = pmid_el.text if pmid_el is not None else ""
        # ArticleTitle may contain sub-elements (e.g. <i>); collect all text
        title = "".join(title_el.itertext()) if title_el is not None else "Unknown Title"
        abstract = abstract_el.text if abstract_el is not None else ""

        papers.append({
            "id": pmid,
            "pmid": pmid,
            "title": title,
            "authors": authors if authors else ["Unknown"],
            "summary": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "published": year,
            "source": "pubmed",
        })
    return papers


@tool
def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """
    Search PubMed for biomedical literature via NCBI E-utilities.

    Returns title, abstract, PMID, URL, and author list per result.
    Use for clinical and biomedical questions.

    Args:
        query: Search terms (e.g., "CAR-T cell lymphoma clinical trial")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of papers with keys: id, pmid, title, authors, summary, url, published, source
    """
    executor = _get_executor()
    future = executor.submit(_fetch_pubmed_results, query, max_results)
    return future.result()