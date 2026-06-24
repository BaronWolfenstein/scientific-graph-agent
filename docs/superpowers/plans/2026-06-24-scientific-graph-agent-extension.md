# Scientific Graph Agent Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `scientific-graph-agent` LangGraph app to use Anthropic/Claude instead of OpenAI, PubMed instead of ArXiv, a two-audience Pydantic-validated structured-output node, and a HITL approval gate before finalizing outputs.

**Architecture:** A new `create_demo_graph()` factory in `graph.py` wires a linear pipeline: clarifier → pubmed_researcher → summarizer → dual_audience → hitl_approval → END. A central LLM factory (`llm.py`) replaces the scattered `ChatOpenAI` instantiations. PubMed retrieval lives in `tools.py` as a drop-in replacement for the ArXiv tool with the same paper-dict contract. Two Pydantic schemas (`schemas.py`) define the structured outputs; their node validates + retries once on schema failure. The HITL node uses LangGraph's `interrupt()` pattern, already established by the existing `approver_node`.

**Tech Stack:** `langchain-anthropic` (`ChatAnthropic`), NCBI E-utilities via `urllib` (stdlib, no new deps), Pydantic v2 (`model_validate_json`), LangGraph `interrupt()` + `SqliteSaver` checkpointer, `pytest`.

## Global Constraints

- Model string: `claude-sonnet-4-6` — never use any other model string.
- Max tokens: `max_tokens=1000` on every `ChatAnthropic` instantiation.
- API key: `os.environ["ANTHROPIC_API_KEY"]` — never hard-coded.
- Checkpointer pins: `langgraph-checkpoint-sqlite >= 3.0.1`, `langgraph-checkpoint >= 4.0.1` — do not use lower versions.
- Minimal changes: do not rewrite existing nodes, do not add features beyond Steps 1–5.
- Match existing patterns: paper dicts use keys `id`, `title`, `authors`, `summary`, `url`, `published`, `source`, `relevance_score`; PubMed adds `pmid`.
- No secrets in code. Env vars only.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Modify | Add `langchain-anthropic`, bump checkpointer pins, add `biopython` (unused but spec-mentioned; keep via requests instead) |
| `src/agent_graph/llm.py` | Create | `get_llm(temperature)` factory — single LLM construction site |
| `src/agent_graph/schemas.py` | Create | `Evidence`, `ClinicianSummary`, `TechnicalSummary` Pydantic models |
| `src/agent_graph/tools.py` | Modify | Add `search_pubmed` tool + `_fetch_pubmed_results` helper |
| `src/agent_graph/state.py` | Modify | Replace `ChatOpenAI` import with `get_llm`; add `clinician_summary`, `technical_summary` to `OutputState`/`InternalState` |
| `src/agent_graph/nodes.py` | Modify | Replace all `ChatOpenAI(...)` with `get_llm(...)`; add `pubmed_researcher_node`; add `dual_audience_node`; add `hitl_approval_node` + `route_after_hitl` |
| `src/agent_graph/graph.py` | Modify | Add `create_demo_graph()` |
| `run_demo.py` | Create | CLI entry point: one query end-to-end with HITL |
| `tests/test_pubmed.py` | Create | PubMed tool unit tests |
| `tests/test_schemas.py` | Create | Schema validation + retry-pattern unit tests |
| `EXTENSION_NOTES.md` | Create | Honest record of extension changes |

---

### Task 1: Anthropic dependency + LLM factory + swap all ChatOpenAI

**Files:**
- Modify: `pyproject.toml`
- Create: `src/agent_graph/llm.py`
- Modify: `src/agent_graph/state.py`
- Modify: `src/agent_graph/nodes.py`

**Interfaces:**
- Produces: `get_llm(temperature: float = 0.0) -> ChatAnthropic` — imported by Tasks 3, 4 nodes and state.py

- [ ] **Step 1: Update pyproject.toml**

Replace the deps block:

```toml
[project]
name = "scientific-graph-agent"
version = "0.1.0"
description = "Simple agent graph for scientific paper exploration with memory"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=1.0.1",
    "langgraph-prebuilt>=1.0.1",
    "langgraph-sdk>=0.2.9",
    "langgraph-checkpoint-sqlite>=3.0.1",
    "langgraph-checkpoint>=4.0.1",
    "langgraph-cli[inmem]>=0.4.12",
    "langgraph-api>=0.7.0",
    "langsmith>=0.4.37",
    "langchain-community>=0.4",
    "langchain-core>=1.0.0",
    "langchain-openai>=1.0.0",
    "langchain-anthropic>=0.3.0",
    "anthropic>=0.40.0",
    "arxiv>=2.2.0",
    "wikipedia>=1.4.0",
    "python-dotenv>=1.1.1",
    "jupyter>=1.1.1",
    "ipykernel>=7.0.1",
    "langchain>=1.0.3",
    "aiosqlite",
    "trustcall>=0.0.39",
    "pytest>=8.0.0",
]
```

- [ ] **Step 2: Install updated deps**

```bash
cd ~/git/scientific-graph-agent
pip install -e ".[dev]" 2>/dev/null || pip install -e .
# Verify anthropic installed
python3 -c "from langchain_anthropic import ChatAnthropic; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Write the failing test**

Create `tests/test_llm_factory.py`:

```python
import os
import pytest
from unittest.mock import patch

def test_get_llm_returns_chat_anthropic():
    from agent_graph.llm import get_llm
    from langchain_anthropic import ChatAnthropic
    llm = get_llm()
    assert isinstance(llm, ChatAnthropic)

def test_get_llm_uses_correct_model():
    from agent_graph.llm import get_llm
    llm = get_llm()
    assert llm.model == "claude-sonnet-4-6"

def test_get_llm_max_tokens():
    from agent_graph.llm import get_llm
    llm = get_llm()
    assert llm.max_tokens == 1000

def test_get_llm_temperature_respected():
    from agent_graph.llm import get_llm
    llm = get_llm(temperature=0.7)
    assert llm.temperature == 0.7

def test_get_llm_no_hardcoded_key():
    """API key must come from env, not be embedded in the factory."""
    import inspect
    import agent_graph.llm as llm_module
    src = inspect.getsource(llm_module)
    assert "sk-ant" not in src
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd ~/git/scientific-graph-agent
python3 -m pytest tests/test_llm_factory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_graph.llm'`

- [ ] **Step 5: Create `src/agent_graph/llm.py`**

```python
"""Central LLM factory — single construction site for the Anthropic client."""
import os
from langchain_anthropic import ChatAnthropic


def get_llm(temperature: float = 0.0) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        temperature=temperature,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python3 -m pytest tests/test_llm_factory.py -v
```

Expected: 5 passed

- [ ] **Step 7: Update `src/agent_graph/state.py` — remove ChatOpenAI import, use get_llm**

The `summarize_messages` function currently does `llm = ChatOpenAI(model=llm_model_name, temperature=llm_temperature)`. Replace with `get_llm(temperature=llm_temperature)`.

Full updated `src/agent_graph/state.py`:

```python
"""Shared state between graph nodes."""
from typing import TypedDict, List, Annotated
from typing import NotRequired
from operator import add
from langchain_core.messages import BaseMessage, SystemMessage

from agent_graph.llm import get_llm

# === REDUCERS ===

def take_max(left: int, right: int) -> int:
    """Reducer that takes the maximum of two values."""
    return max(left, right)

def keep_top_papers(existing: List[dict], new: List[dict], n_top: int = 10) -> List[dict]:
    """
    Reducer function to maintain top N papers by relevance score.
    """
    if isinstance(new, list) and len(new) == 0 and len(existing) > 0:
        return []
    
    combined = existing + new
    
    seen = {}
    for paper in combined:
        paper_id = paper['id']
        if paper_id not in seen or paper['relevance_score'] > seen[paper_id]['relevance_score']:
            seen[paper_id] = paper
    
    sorted_papers = sorted(seen.values(), 
                          key=lambda p: p.get('relevance_score', 0), 
                          reverse=True)
    return sorted_papers[:n_top]

def summarize_messages(
        existing: List[BaseMessage],
        new: List[BaseMessage],
        min_message: int = 3,
        llm_model_name: str = "claude-sonnet-4-6",
        llm_temperature: int = 1,
        ) -> List[BaseMessage]:
    """
    Reducer function to summarize conversation history when it gets too long.
    """
    combined = existing + new

    if len(combined) <= min_message:
        return combined

    old_messages = combined[:-min_message]
    recent_messages = combined[-min_message:]

    llm = get_llm(temperature=llm_temperature)

    messages_to_summarize = [
        SystemMessage(content="Summarize this conversation history concisely in 2-3 sentences.")
    ] + old_messages

    summary_response = llm.invoke(messages_to_summarize)
    summary_text = summary_response.content

    return [SystemMessage(content=summary_text)] + recent_messages

# === PUBLIC INTERFACE ===

class InputState(TypedDict):
    """Input state for the graph - user-provided query."""
    query: str
    llm_temperature: NotRequired[float]
    max_papers: NotRequired[int]
    max_iterations: NotRequired[int]
    num_queries: NotRequired[int]

class OutputState(TypedDict):
    """Output state for the graph - results returned to user."""
    summary: str
    papers: Annotated[List[dict], lambda e, n: keep_top_papers(e, n, n_top=10)]
    messages: Annotated[List[BaseMessage], lambda e, n: summarize_messages(
        e, n,
        min_message=3,
        llm_model_name="claude-sonnet-4-6",
        llm_temperature=1)]
    clinician_summary: NotRequired[dict]
    technical_summary: NotRequired[dict]

# === INTERNAL WORKING STATE ===

class InternalState(InputState, OutputState):
    """Full state used internally by nodes."""
    refined_query: NotRequired[str]
    refined_queries: NotRequired[List[str]]
    iteration: Annotated[int, take_max]
    approved: NotRequired[bool]
    
class PrivateState(TypedDict):
    """Private state for internal node processing."""
    refined_query: NotRequired[str]
    refined_queries: NotRequired[List[str]]
    iteration: int
```

Note: `llm_model` is removed from `InputState` since it's no longer meaningful — the factory always uses `claude-sonnet-4-6`. The `llm_model` key in existing nodes will fall back to the factory default.

- [ ] **Step 8: Update `src/agent_graph/nodes.py` — replace all ChatOpenAI with get_llm**

The file has `ChatOpenAI` at lines 3 (import) and instantiated inside: `clarifier_node` (line ~54,56), `arxiv_researcher_node` (line ~176), `wikipedia_researcher_node` (line ~241), `summarizer_node` (line ~304), `arxiv_researcher_node_streaming` (line ~417), `wikipedia_researcher_node_streaming` (line ~515), `summarizer_node_streaming` (line ~582).

Replace the import at the top:

```python
# REMOVE this line:
from langchain_openai import ChatOpenAI

# ADD this line (alongside existing imports):
from agent_graph.llm import get_llm
```

Replace every `ChatOpenAI` instantiation — there are 7 total:

1. In `clarifier_node` (~line 54):
   ```python
   # REMOVE:
   llm = ChatOpenAI(
       model=state.get("llm_model", "gpt-4o-mini"),
       temperature=state.get("llm_temperature", 0.3 if num_queries > 1 else 0)
   )
   # REPLACE WITH:
   llm = get_llm(temperature=state.get("llm_temperature", 0.3 if num_queries > 1 else 0))
   ```

2. In `arxiv_researcher_node` (~line 176):
   ```python
   # REMOVE:
   llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
   # REPLACE WITH:
   llm = get_llm(temperature=llm_temperature)
   ```
   Also remove the lines that read `llm_model = state.get("llm_model", "gpt-4o-mini")` since the model is no longer configurable.

3. In `wikipedia_researcher_node` (~line 241):
   ```python
   # REMOVE:
   llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
   # REPLACE WITH:
   llm = get_llm(temperature=llm_temperature)
   ```
   Same: remove `llm_model = state.get(...)` lines.

4. In `summarizer_node` (~line 304):
   ```python
   # REMOVE:
   llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
   # REPLACE WITH:
   llm = get_llm(temperature=llm_temperature)
   ```
   Remove `llm_model` read.

5. In `arxiv_researcher_node_streaming` (~line 417):
   ```python
   # REMOVE:
   llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
   # REPLACE WITH:
   llm = get_llm(temperature=llm_temperature)
   ```

6. In `wikipedia_researcher_node_streaming` (~line 515):
   ```python
   # REMOVE:
   llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
   # REPLACE WITH:
   llm = get_llm(temperature=llm_temperature)
   ```

7. In `summarizer_node_streaming` (~line 582):
   ```python
   # REMOVE:
   llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
   # REPLACE WITH:
   llm = get_llm(temperature=llm_temperature)
   ```

- [ ] **Step 9: Verify nodes.py imports cleanly**

```bash
cd ~/git/scientific-graph-agent
python3 -c "from agent_graph.nodes import clarifier_node; print('OK')"
```

Expected: `OK` (no ImportError)

- [ ] **Step 10: Commit**

```bash
cd ~/git/scientific-graph-agent
git add pyproject.toml src/agent_graph/llm.py src/agent_graph/state.py src/agent_graph/nodes.py tests/test_llm_factory.py
git commit -m "feat: swap OpenAI → Anthropic claude-sonnet-4-6 via central get_llm factory"
```

---

### Task 2: PubMed retrieval tool + pubmed_researcher_node

**Files:**
- Modify: `src/agent_graph/tools.py`
- Modify: `src/agent_graph/nodes.py`
- Create: `tests/test_pubmed.py`

**Interfaces:**
- Consumes: `get_llm` from Task 1
- Produces: `search_pubmed` LangChain `@tool` returning `list[dict]` with keys `id`, `pmid`, `title`, `authors`, `summary`, `url`, `published`, `source`
- Produces: `pubmed_researcher_node(state: InternalState) -> OutputState` — added to `nodes.py` and `TOOL_REGISTRY`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pubmed.py`:

```python
"""Tests for PubMed retrieval tool — uses real NCBI API."""
import pytest


def test_fetch_pubmed_results_returns_list():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("CAR-T cell lymphoma", max_results=2)
    assert isinstance(results, list)
    assert len(results) > 0


def test_pubmed_paper_has_required_fields():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("immunotherapy cancer", max_results=1)
    assert len(results) == 1
    paper = results[0]
    for key in ("id", "pmid", "title", "authors", "summary", "url", "published", "source"):
        assert key in paper, f"Missing key: {key}"


def test_pubmed_url_format():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("aspirin cardiology", max_results=1)
    assert len(results) == 1
    pmid = results[0]["pmid"]
    assert results[0]["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def test_pubmed_source_field():
    from agent_graph.tools import _fetch_pubmed_results
    results = _fetch_pubmed_results("diabetes mellitus treatment", max_results=1)
    assert results[0]["source"] == "pubmed"


def test_search_pubmed_tool_is_callable():
    from agent_graph.tools import search_pubmed
    # Verify it's a LangChain tool (has .invoke)
    assert hasattr(search_pubmed, "invoke")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/git/scientific-graph-agent
python3 -m pytest tests/test_pubmed.py -v
```

Expected: FAIL with `ImportError: cannot import name '_fetch_pubmed_results' from 'agent_graph.tools'`

- [ ] **Step 3: Add PubMed retrieval to `src/agent_graph/tools.py`**

Add to the top of the file (after existing imports):

```python
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
```

Add the helper and tool after the Wikipedia tools (before the end of the file):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_pubmed.py -v
```

Expected: 5 passed (these hit the real NCBI API — they should pass; NCBI has generous rate limits for unauthenticated requests)

- [ ] **Step 5: Add `pubmed_researcher_node` to `src/agent_graph/nodes.py`**

First, add the import at the top of `nodes.py`:

```python
# Add to existing import line:
from agent_graph.tools import search_arxiv, search_arxiv_streaming, search_wikipedia, search_wikipedia_streaming, search_pubmed
```

Add `"pubmed"` to `TOOL_REGISTRY` (after line ~40):

```python
TOOL_REGISTRY = {
    "arxiv": {
        "sync": search_arxiv,
        "async": search_arxiv_streaming,
        "description": "ArXiv scientific papers",
        "param_name": "query"
    },
    "wikipedia": {
        "sync": search_wikipedia,
        "async": search_wikipedia_streaming,
        "description": "Wikipedia articles",
        "param_name": "topic"
    },
    "pubmed": {
        "sync": search_pubmed,
        "async": None,
        "description": "PubMed biomedical literature",
        "param_name": "query"
    }
}
```

Add the node function after `wikipedia_researcher_node` (before the `summarizer_node`):

```python
def pubmed_researcher_node(state: InternalState) -> OutputState:
    """PubMed researcher node: searches PubMed and scores paper relevance."""
    max_papers = state.get("max_papers", 5)
    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"Searching PubMed: '{query}' (iteration {iteration})")
    papers = search_pubmed.invoke({"query": query, "max_results": max_papers})
    logging.info(f"Found {len(papers)} PubMed papers")

    llm = get_llm(temperature=state.get("llm_temperature", 0))

    scored_papers = []
    for paper in papers:
        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator.
Score how relevant a paper is to the user's query on a scale from 1 to 100.
Consider: direct relevance, depth of content, usefulness for answering the query.
Respond with ONLY a number between 1 and 100, nothing else."""),
            HumanMessage(content=f"""User Query: {original_query}

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}
Abstract: {paper['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]
        response = llm.invoke(score_messages)
        try:
            relevance_score = int(response.content.strip())
        except ValueError:
            relevance_score = 50
        paper['relevance_score'] = max(1, min(100, relevance_score))
        scored_papers.append(paper)
        logging.info(f"  📄 {paper['title'][:60]}... - Score: {paper['relevance_score']}")

    return {
        "papers": scored_papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"Found {len(scored_papers)} papers on PubMed for query: {query}",
                name="PubMedResearcher"
            )
        ]
    }
```

- [ ] **Step 6: Verify node imports cleanly**

```bash
python3 -c "from agent_graph.nodes import pubmed_researcher_node; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/agent_graph/tools.py src/agent_graph/nodes.py tests/test_pubmed.py
git commit -m "feat: add PubMed retrieval tool and pubmed_researcher_node via NCBI E-utilities"
```

---

### Task 3: Pydantic schemas + two-audience structured-output node

**Files:**
- Create: `src/agent_graph/schemas.py`
- Modify: `src/agent_graph/nodes.py` (add `dual_audience_node`, `_extract_json`, `_generate_with_retry`)
- Create: `tests/test_schemas.py`

**Interfaces:**
- Consumes: `get_llm` from Task 1; `InternalState` with `papers` (list of dicts with `pmid`) and `summary` fields
- Produces: `dual_audience_node(state: InternalState) -> dict` returning `{"clinician_summary": dict, "technical_summary": dict}` where dicts are `.model_dump()` of the Pydantic models
- Produces: `ClinicianSummary`, `TechnicalSummary`, `Evidence` importable from `agent_graph.schemas`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas.py`:

```python
"""Tests for Pydantic schemas and the JSON extraction + retry helpers."""
import json
import pytest
from pydantic import ValidationError


def test_evidence_model_valid():
    from agent_graph.schemas import Evidence
    e = Evidence(claim="Drug X reduces mortality", pmid="12345678",
                 source_url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    assert e.pmid == "12345678"


def test_clinician_summary_valid():
    from agent_graph.schemas import ClinicianSummary, Evidence
    cs = ClinicianSummary(
        bottom_line="CAR-T shows durable remission in DLBCL.",
        key_findings=["60% CR rate at 12 months"],
        evidence=[Evidence(claim="CR rate", pmid="34567890",
                           source_url="https://pubmed.ncbi.nlm.nih.gov/34567890/")],
        confidence_note="Based on 2 RCTs."
    )
    assert cs.audience == "clinician"


def test_technical_summary_valid():
    from agent_graph.schemas import TechnicalSummary, Evidence
    ts = TechnicalSummary(
        detailed_findings="CD19-directed CAR-T achieved 58% ORR.",
        methodology_notes="Phase II, n=93, LBCL ≥ 2 prior lines.",
        evidence=[Evidence(claim="ORR", pmid="34567890",
                           source_url="https://pubmed.ncbi.nlm.nih.gov/34567890/")],
        caveats=["Short follow-up", "Single arm"]
    )
    assert ts.audience == "technical"


def test_clinician_summary_missing_required_field_raises():
    from agent_graph.schemas import ClinicianSummary
    with pytest.raises(ValidationError):
        ClinicianSummary(key_findings=[], evidence=[], confidence_note="ok")
        # missing bottom_line


def test_extract_json_strips_markdown_fence():
    from agent_graph.nodes import _extract_json
    raw = '```json\n{"key": "value"}\n```'
    assert _extract_json(raw) == '{"key": "value"}'


def test_extract_json_passthrough_plain():
    from agent_graph.nodes import _extract_json
    raw = '{"key": "value"}'
    assert _extract_json(raw) == '{"key": "value"}'


def test_generate_with_retry_succeeds_first_attempt(monkeypatch):
    from agent_graph.nodes import _generate_with_retry
    from agent_graph.schemas import ClinicianSummary, Evidence
    from langchain_core.messages import SystemMessage

    valid_json = json.dumps({
        "bottom_line": "Drug reduces mortality.",
        "key_findings": ["50% reduction"],
        "evidence": [{"claim": "reduction", "pmid": "111",
                      "source_url": "https://pubmed.ncbi.nlm.nih.gov/111/"}],
        "confidence_note": "Single RCT."
    })

    class FakeLLM:
        def invoke(self, messages):
            class Resp:
                content = valid_json
            return Resp()

    result, first_err = _generate_with_retry(FakeLLM(), [SystemMessage(content="test")], ClinicianSummary)
    assert isinstance(result, ClinicianSummary)
    assert first_err is None


def test_generate_with_retry_retries_on_bad_json(monkeypatch):
    from agent_graph.nodes import _generate_with_retry
    from agent_graph.schemas import ClinicianSummary
    from langchain_core.messages import SystemMessage

    valid_json = json.dumps({
        "bottom_line": "Drug reduces mortality.",
        "key_findings": ["50% reduction"],
        "evidence": [{"claim": "reduction", "pmid": "111",
                      "source_url": "https://pubmed.ncbi.nlm.nih.gov/111/"}],
        "confidence_note": "Single RCT."
    })

    call_count = {"n": 0}

    class FakeLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            class Resp:
                content = "not valid json" if call_count["n"] == 1 else valid_json
            return Resp()

    result, first_err = _generate_with_retry(FakeLLM(), [SystemMessage(content="test")], ClinicianSummary)
    assert isinstance(result, ClinicianSummary)
    assert first_err is not None  # first attempt failed
    assert call_count["n"] == 2   # retried exactly once
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_graph.schemas'` (and `_extract_json` not found)

- [ ] **Step 3: Create `src/agent_graph/schemas.py`**

```python
"""Pydantic output schemas for the two-audience structured-output node."""
from pydantic import BaseModel, Field
from typing import Literal


class Evidence(BaseModel):
    claim: str
    pmid: str
    source_url: str


class ClinicianSummary(BaseModel):
    audience: Literal["clinician"] = "clinician"
    bottom_line: str = Field(..., description="One-sentence actionable takeaway")
    key_findings: list[str]
    evidence: list[Evidence]
    confidence_note: str = Field(..., description="What is and isn't well-supported")


class TechnicalSummary(BaseModel):
    audience: Literal["technical"] = "technical"
    detailed_findings: str
    methodology_notes: str
    evidence: list[Evidence]
    caveats: list[str]
```

- [ ] **Step 4: Add `_extract_json`, `_generate_with_retry`, and `dual_audience_node` to `src/agent_graph/nodes.py`**

Add the import at the top of `nodes.py` (alongside existing imports):

```python
import json as _json
from pydantic import ValidationError
from langchain_core.messages import AIMessage as _AIMessage  # already imported as AIMessage
from agent_graph.schemas import ClinicianSummary, TechnicalSummary, Evidence
```

(Note: `AIMessage` is already imported — don't duplicate. Just add `json`, `ValidationError`, and the schema imports.)

Add the two helpers and the node function after `route_after_approval` at the end of the file:

```python
# ============================================================================
# JSON HELPERS FOR STRUCTURED OUTPUT
# ============================================================================

def _extract_json(text: str) -> str:
    """Strip markdown code fence (```json ... ```) if present, return bare JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop opening fence line and closing fence line
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    return text.strip()


def _generate_with_retry(llm, messages: list, schema_cls) -> tuple:
    """
    Prompt llm with messages, validate response against schema_cls.
    On ValidationError, re-prompts once with the error fed back.

    Returns:
        (instance, first_error_str_or_None)
    """
    response = llm.invoke(messages)
    raw = response.content
    json_str = _extract_json(raw)
    try:
        return schema_cls.model_validate_json(json_str), None
    except (ValidationError, Exception) as e:
        first_err = str(e)
        retry_messages = messages + [
            AIMessage(content=raw),
            HumanMessage(
                content=(
                    f"The JSON you returned has validation errors:\n{first_err}\n\n"
                    "Please fix ALL errors and return ONLY valid JSON conforming to the schema. "
                    "No markdown fences, no explanation."
                ),
                name="User"
            )
        ]
        retry_response = llm.invoke(retry_messages)
        retry_json = _extract_json(retry_response.content)
        validated = schema_cls.model_validate_json(retry_json)
        return validated, first_err


# ============================================================================
# DUAL-AUDIENCE STRUCTURED-OUTPUT NODE
# ============================================================================

def dual_audience_node(state: InternalState) -> dict:
    """
    Generates two Pydantic-validated summaries from retrieved papers:
      - ClinicianSummary: actionable bottom-line for clinicians
      - TechnicalSummary: detailed methodology + caveats for researchers

    Retries once per schema on validation failure (schema-failure retry pattern).
    Only cites PMIDs present in the retrieved papers (grounding constraint).
    """
    papers = state.get("papers", [])
    query = state["query"]

    llm = get_llm(temperature=0)

    # Build grounded paper context — list PMIDs explicitly so the model knows which to cite
    papers_context = "\n\n".join([
        f"PMID: {p.get('pmid', p['id'])}\n"
        f"Title: {p['title']}\n"
        f"Authors: {', '.join(p['authors'][:3])}\n"
        f"Published: {p['published']}\n"
        f"Abstract: {p['summary'][:500]}\n"
        f"URL: {p['url']}"
        for p in papers
    ])

    allowed_pmids = [p.get("pmid", p["id"]) for p in papers]
    allowed_str = ", ".join(allowed_pmids)

    grounding_rule = (
        f"GROUNDING RULE: You MUST only cite PMIDs from this set: [{allowed_str}]. "
        "If evidence is insufficient for a claim, say so explicitly. "
        "Do NOT invent PMIDs or URLs."
    )

    # --- Clinician summary ---
    clinician_schema_str = """{
  "audience": "clinician",
  "bottom_line": "<one-sentence actionable takeaway>",
  "key_findings": ["<finding 1>", "..."],
  "evidence": [{"claim": "<claim>", "pmid": "<pmid>", "source_url": "https://pubmed.ncbi.nlm.nih.gov/<pmid>/"}],
  "confidence_note": "<what is and isn't well-supported>"
}"""

    clinician_messages = [
        SystemMessage(content=(
            f"You are a clinical evidence synthesizer writing for a treating physician.\n"
            f"{grounding_rule}\n"
            "Return ONLY valid JSON matching this schema (no markdown fence):\n"
            f"{clinician_schema_str}"
        )),
        HumanMessage(content=(
            f"Clinical question: {query}\n\n"
            f"Retrieved papers:\n{papers_context}\n\n"
            "Generate the clinician summary JSON."
        ), name="User")
    ]

    clinician_result, clinician_retry_err = _generate_with_retry(
        llm, clinician_messages, ClinicianSummary
    )
    if clinician_retry_err:
        logging.warning(f"ClinicianSummary required retry: {clinician_retry_err[:100]}")

    # --- Technical summary ---
    technical_schema_str = """{
  "audience": "technical",
  "detailed_findings": "<detailed findings paragraph>",
  "methodology_notes": "<study design, N, endpoints, methods>",
  "evidence": [{"claim": "<claim>", "pmid": "<pmid>", "source_url": "https://pubmed.ncbi.nlm.nih.gov/<pmid>/"}],
  "caveats": ["<caveat 1>", "..."]
}"""

    technical_messages = [
        SystemMessage(content=(
            f"You are a research methodologist writing for a clinical scientist or statistician.\n"
            f"{grounding_rule}\n"
            "Return ONLY valid JSON matching this schema (no markdown fence):\n"
            f"{technical_schema_str}"
        )),
        HumanMessage(content=(
            f"Research question: {query}\n\n"
            f"Retrieved papers:\n{papers_context}\n\n"
            "Generate the technical summary JSON."
        ), name="User")
    ]

    technical_result, technical_retry_err = _generate_with_retry(
        llm, technical_messages, TechnicalSummary
    )
    if technical_retry_err:
        logging.warning(f"TechnicalSummary required retry: {technical_retry_err[:100]}")

    logging.info("✅ Dual-audience summaries generated")

    return {
        "clinician_summary": clinician_result.model_dump(),
        "technical_summary": technical_result.model_dump(),
        "messages": [
            AIMessage(
                content="Generated clinician + technical structured summaries.",
                name="DualAudience"
            )
        ]
    }
```

- [ ] **Step 5: Run schema tests to verify they pass**

```bash
python3 -m pytest tests/test_schemas.py -v
```

Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/agent_graph/schemas.py src/agent_graph/nodes.py tests/test_schemas.py
git commit -m "feat: add Pydantic schemas and dual_audience_node with schema-failure retry"
```

---

### Task 4: HITL approval node + demo graph + entry point + docs

**Files:**
- Modify: `src/agent_graph/nodes.py` (add `hitl_approval_node`, `route_after_hitl`)
- Modify: `src/agent_graph/graph.py` (add `create_demo_graph`)
- Create: `run_demo.py`
- Create: `EXTENSION_NOTES.md`

**Interfaces:**
- Consumes: `dual_audience_node` output (`clinician_summary`, `technical_summary` in state); `interrupt()` LangGraph primitive
- Produces: `hitl_approval_node` sets `approved: True/False` in state; `route_after_hitl` returns `"end"`
- Produces: `create_demo_graph()` returns compiled graph with SQLite checkpointer + interrupt inside `hitl_approval_node`

- [ ] **Step 1: Add `hitl_approval_node` and `route_after_hitl` to `src/agent_graph/nodes.py`**

Append after `dual_audience_node`:

```python
# ============================================================================
# HITL APPROVAL GATE
# ============================================================================

def hitl_approval_node(state: InternalState) -> dict:
    """
    Interrupts the graph so a human can review both draft summaries before
    they are finalized. Uses LangGraph's interrupt() — graph resumes when
    Command(resume={"action": "approve"|"reject"}) is passed to invoke().
    """
    cs = state.get("clinician_summary", {})
    ts = state.get("technical_summary", {})

    def _fmt_clinician(cs: dict) -> str:
        lines = [
            "── CLINICIAN SUMMARY ─────────────────────────────────",
            f"Bottom line : {cs.get('bottom_line', '')}",
            "",
            "Key findings:",
        ]
        for f in cs.get("key_findings", []):
            lines.append(f"  • {f}")
        lines += [
            "",
            f"Confidence : {cs.get('confidence_note', '')}",
            "",
            "Evidence cited:",
        ]
        for e in cs.get("evidence", []):
            lines.append(f"  [{e.get('pmid','')}] {e.get('claim','')} — {e.get('source_url','')}")
        return "\n".join(lines)

    def _fmt_technical(ts: dict) -> str:
        lines = [
            "── TECHNICAL SUMMARY ─────────────────────────────────",
            ts.get("detailed_findings", ""),
            "",
            f"Methodology : {ts.get('methodology_notes', '')}",
            "",
            "Caveats:",
        ]
        for c in ts.get("caveats", []):
            lines.append(f"  • {c}")
        lines += ["", "Evidence cited:"]
        for e in ts.get("evidence", []):
            lines.append(f"  [{e.get('pmid','')}] {e.get('claim','')} — {e.get('source_url','')}")
        return "\n".join(lines)

    display = _fmt_clinician(cs) + "\n\n" + _fmt_technical(ts)

    decision = interrupt({
        "type": "summary_approval",
        "display": display,
        "message": (
            "\nReview the draft summaries above.\n"
            "  approve — finalize and return both summaries\n"
            "  reject  — discard summaries and end\n"
        )
    })

    action = decision.get("action", "reject") if isinstance(decision, dict) else str(decision)

    if action == "approve":
        logging.info("✅ Summaries approved by reviewer")
        return {
            "approved": True,
            "messages": [AIMessage(content="✅ Summaries approved", name="HITL")]
        }

    logging.info("❌ Summaries rejected by reviewer")
    return {
        "approved": False,
        "summary": "[REJECTED BY REVIEWER — summaries not finalized]",
        "messages": [AIMessage(content="❌ Summaries rejected", name="HITL")]
    }


def route_after_hitl(state: InternalState) -> str:
    return "end"
```

- [ ] **Step 2: Add `create_demo_graph` to `src/agent_graph/graph.py`**

Add imports at the top of `graph.py` (alongside existing node imports):

```python
from agent_graph.nodes import (
    clarifier_node,
    arxiv_researcher_node,
    wikipedia_researcher_node,
    arxiv_researcher_node_streaming,
    wikipedia_researcher_node_streaming,
    summarizer_node,
    should_continue,
    summarizer_node_streaming,
    approver_node,
    route_after_approval,
    pubmed_researcher_node,     # NEW
    dual_audience_node,         # NEW
    hitl_approval_node,         # NEW
    route_after_hitl,           # NEW
)
```

Add the factory function at the end of `graph.py`:

```python
def create_demo_graph():
    """
    Demo graph for Thursday interview:
      clarifier → pubmed_researcher → summarizer → dual_audience → hitl_approval → END

    The HITL interrupt fires inside hitl_approval_node via interrupt().
    Resume with Command(resume={"action": "approve"}) or Command(resume={"action": "reject"}).

    Checkpointer: SQLite in-memory, pinned to langgraph-checkpoint-sqlite>=3.0.1.
    """
    workflow = StateGraph(
        InternalState,
        input=InputState,
        output=OutputState,
    )

    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("pubmed_researcher", pubmed_researcher_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("dual_audience", dual_audience_node)
    workflow.add_node("hitl_approval", hitl_approval_node)

    workflow.set_entry_point("clarifier")
    workflow.add_edge("clarifier", "pubmed_researcher")
    workflow.add_edge("pubmed_researcher", "summarizer")
    workflow.add_edge("summarizer", "dual_audience")
    workflow.add_edge("dual_audience", "hitl_approval")
    workflow.add_conditional_edges(
        "hitl_approval",
        route_after_hitl,
        {"end": END}
    )

    import sqlite3
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)
```

- [ ] **Step 3: Verify graph compiles without errors**

```bash
cd ~/git/scientific-graph-agent
python3 -c "
from agent_graph.graph import create_demo_graph
g = create_demo_graph()
print('Nodes:', list(g.nodes))
"
```

Expected output (order may vary):
```
Nodes: ['__start__', 'clarifier', 'pubmed_researcher', 'summarizer', 'dual_audience', 'hitl_approval', '__end__']
```

- [ ] **Step 4: Create `run_demo.py`**

```python
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
            for intr in task.interrupts:
                interrupt_payload = intr.value
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
    if result.get("approved") is False or result.get("summary", "").startswith("[REJECTED"):
        print("\n❌ Summaries rejected — pipeline ended without output.")
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
```

- [ ] **Step 5: Run end-to-end smoke test**

```bash
cd ~/git/scientific-graph-agent
python3 run_demo.py
```

Expected: pipeline runs, prints draft summaries, pauses for input. Type `approve`. Both final summaries print with PubMed URLs. No crash.

If NCBI is slow, the PubMed search may take 10–15s — that's expected.

- [ ] **Step 6: Create `EXTENSION_NOTES.md`**

```markdown
# Extension Notes

This file records exactly what was changed from the original `scientific-graph-agent` 
by Lina Faik. I adapted and instrumented the existing framework — I did not author it.

## Changes made (2026-06-24)

### 1. Anthropic backend (Step 1)
- Added `langchain-anthropic>=0.3.0` and `anthropic>=0.40.0` to `pyproject.toml`
- Created `src/agent_graph/llm.py`: `get_llm(temperature)` factory returning
  `ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000, ...)`
- Replaced all `ChatOpenAI(...)` instantiations in `nodes.py` (~7 sites) and
  `state.py` (`summarize_messages` reducer) with `get_llm(...)`
- API key read from `ANTHROPIC_API_KEY` env var; never hard-coded

### 2. PubMed retrieval (Step 2)
- Added `_fetch_pubmed_results` helper and `search_pubmed` LangChain `@tool`
  to `src/agent_graph/tools.py`
- Uses NCBI E-utilities (`esearch.fcgi` + `efetch.fcgi`) via stdlib `urllib` —
  no extra dependencies
- Returns the same paper-dict contract as the ArXiv tool
  (`id`, `title`, `authors`, `summary`, `url`, `published`, `source`) plus `pmid`
- Added `pubmed_researcher_node` to `nodes.py` and `"pubmed"` to `TOOL_REGISTRY`
- ArXiv tool and nodes remain intact; PubMed is additive

### 3. Two-audience structured-output node with schema-failure retry (Step 3)
- Created `src/agent_graph/schemas.py`: `Evidence`, `ClinicianSummary`,
  `TechnicalSummary` Pydantic v2 models
- Added `_extract_json` (strips markdown fences) and `_generate_with_retry`
  to `nodes.py`: prompts model, validates with `model_validate_json()`,
  re-prompts once with the validation error on failure (the retry is demonstrable
  and tested in `tests/test_schemas.py`)
- Added `dual_audience_node` to `nodes.py`: generates both summaries with
  grounding constraint (only cite PMIDs from retrieved papers)
- Added `clinician_summary` and `technical_summary` fields to `OutputState`
  and `InternalState` in `state.py`

### 4. HITL approval gate (Step 4)
- Added `hitl_approval_node` and `route_after_hitl` to `nodes.py`
- Uses LangGraph's `interrupt()` inside the node — graph suspends mid-node,
  resumes when `Command(resume={"action": "approve"|"reject"})` is passed
- Pattern follows the existing `approver_node` in the repo
- Gate fires after `dual_audience_node`, before summaries are finalized

### 5. Patched checkpointer (Steps 1, 4)
- Bumped `langgraph-checkpoint-sqlite` to `>=3.0.1` in `pyproject.toml`
- Added `langgraph-checkpoint>=4.0.1`
- Previous pin (`>=3.0.0`) had a known SQLi→RCE chain in older releases

### 6. Demo graph and entry point (Step 4)
- Added `create_demo_graph()` to `graph.py`:
  `clarifier → pubmed_researcher → summarizer → dual_audience → hitl_approval → END`
- Created `run_demo.py`: CLI entry point, handles interrupt/resume loop
- Example query: CAR-T cell therapy in relapsed/refractory DLBCL

## What was NOT changed
- ArXiv and Wikipedia tools and nodes (intact, still usable)
- `create_graph`, `create_streaming_graph`, `create_graph_with_approval`,
  `create_map_reduce_graph` — all original graph factories unchanged
- `state.py` reducers (`keep_top_papers`, `take_max`) — unchanged
- Notebook demos — not updated (use original OpenAI key)
```

- [ ] **Step 7: Run full test suite**

```bash
cd ~/git/scientific-graph-agent
python3 -m pytest tests/ -v
```

Expected: all tests pass (test_llm_factory, test_pubmed, test_schemas). PubMed tests hit the real API — they may be slow (~5s each) but should pass.

- [ ] **Step 8: Final commit**

```bash
git add src/agent_graph/nodes.py src/agent_graph/graph.py run_demo.py EXTENSION_NOTES.md
git commit -m "feat: add HITL approval gate, demo graph, entry point, and extension notes"
```

---

## Acceptance Criteria Checklist

| # | Criterion | Verified by |
|---|-----------|-------------|
| 1 | Runs end-to-end against Anthropic + PubMed | `python3 run_demo.py` |
| 2 | Two-audience node emits Pydantic-validated outputs; retry works | `tests/test_schemas.py::test_generate_with_retry_retries_on_bad_json` |
| 3 | HITL gate pauses before finalizing; respects approve/reject | `run_demo.py` interactive flow |
| 4 | Checkpointer on patched version | `pyproject.toml` pins |
| 5 | `EXTENSION_NOTES.md` accurately lists changes | Human review |
