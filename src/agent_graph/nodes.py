"""Graph nodes: clarifier, researcher, and summarizer."""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt
import asyncio
import json as _json
import jsonschema
from pydantic import ValidationError

from agent_graph.llm import get_llm
from agent_graph.state import InputState, OutputState, PrivateState, InternalState
from agent_graph.tools import search_arxiv, search_arxiv_streaming, search_wikipedia, search_wikipedia_streaming, search_pubmed
from agent_graph.reranker import score_papers, apply_dropoff, final_relevance_score, CANDIDATE_FETCH
from agent_graph.schemas import ClinicianSummary, TechnicalSummary, Evidence


import logging

# Configure the logger (optional but recommended)
logging.basicConfig(
    level=logging.INFO,                     # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# Create a logger instance
logger = logging.getLogger(__name__)

# ============================================================================
# TOOL CONFIGURATION
# ============================================================================

# Default tool mapping
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

# ============================================================================
# SYNC (NON-STREAMING) NODES
# ============================================================================


def clarifier_node(state: InternalState) -> PrivateState:
    """Clarifier node: refines and optimizes the user query."""
    original_query = state["query"]
    conversation_history = state.get("messages", [])
    num_queries = state.get("num_queries", 1)  # Check if multi-query mode is requested

    llm = get_llm(temperature=0)

    logging.info(f"Clarifying query: '{original_query}' (num_queries={num_queries})")

    if num_queries > 1:
        # Multi-query mode for map-reduce
        system_prompt = f"""You are a research assistant specialized in query refinement.
Your task is to transform user questions into {num_queries} diverse and complementary ArXiv search queries.

Guidelines:
- Consider the conversation history to understand context
- Generate {num_queries} different search queries that approach the topic from different angles
- Each query should target different aspects or related concepts
- Use precise technical terminology and domain-specific descriptors rather than generic terms like "survey" or "overview"
- Keep each query concise (5-10 words max)
- Ensure queries are complementary, not redundant
- Do NOT include years, dates, or version numbers — ArXiv treats these as literal search terms and they degrade relevance

Return ONLY the queries, one per line, numbered. Example format:
1. first refined query here
2. second refined query here
3. third refined query here"""

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history for context
        if conversation_history:
            messages.extend(conversation_history)

        # Add the current query
        messages.append(HumanMessage(content=f"Original question: {original_query}", name="User"))

        response = llm.invoke(messages)
        refined_queries_text = response.content.strip()

        # Parse the numbered queries
        lines = refined_queries_text.split('\n')
        refined_queries = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # Remove the number prefix (e.g., "1. ", "2. ")
                query = line.split('.', 1)[1].strip() if '.' in line else line
                refined_queries.append(query)

        # Fallback if parsing fails
        if not refined_queries:
            refined_queries = [original_query]

        logging.info(f"Generated {len(refined_queries)} refined queries:")
        for i, q in enumerate(refined_queries, 1):
            logging.info(f"  {i}. {q}")

        # Return PrivateState fields with multiple queries
        return {
            "refined_queries": refined_queries,
            "refined_query": refined_queries[0],  # Keep first one for backward compatibility
            "iteration": 0,
            "messages": [
                HumanMessage(content=original_query, name="User"),
                AIMessage(content=f"Refined queries:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(refined_queries, 1)), name="Clarifier")
            ]
        }
    else:
        # Original single-query mode
        system_prompt = """You are a research assistant specialized in query refinement.
Your task is to transform user questions into optimal ArXiv search queries.

Guidelines:
- Consider the conversation history to understand context
- If the user asks a follow-up question (e.g., "tell me more about X", "what about Y?"), use the previous context to refine the query
- Identify key scientific concepts and use precise technical terminology
- Keep it concise (5-10 words max) and focus on the core topic
- Use domain-specific descriptors (e.g. "sparse attention", "linear complexity", "state space models") rather than generic terms like "survey" or "overview" which match unrelated papers
- Do NOT include years, dates, or version numbers — ArXiv treats these as literal search terms and they degrade relevance

Return ONLY the refined query, nothing else."""

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history for context
        if conversation_history:
            messages.extend(conversation_history)

        # Add the current query
        messages.append(HumanMessage(content=f"Original question: {original_query}", name="User"))

        response = llm.invoke(messages)
        refined_query = response.content.strip()

        logging.info(f"Refined query: '{refined_query}'")

        # Return PrivateState fields
        return {
            "refined_query": refined_query,
            "iteration": 0,
            "messages": [
                HumanMessage(content=original_query, name="User"),
                AIMessage(content=f"Refined query: {refined_query}", name="Clarifier")
            ]
        }


def arxiv_researcher_node(state: InternalState) -> OutputState:
    """ArXiv researcher node: searches ArXiv for relevant papers and scores their relevance."""
    max_papers = state.get("max_papers", 5)

    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"Searching ArXiv: '{query}' (iteration {iteration})")
    candidates = search_arxiv.invoke({"query": query, "max_results": CANDIDATE_FETCH})
    logging.info(f"Fetched {len(candidates)} candidates, reranking...")

    # Dense + cross-encoder scoring
    candidates = score_papers(original_query, candidates)

    # Drop-off filter: cut at relative score gap, hard ceiling max_k=10
    papers = apply_dropoff(candidates)
    logging.info(f"Drop-off filter kept {len(papers)} / {len(candidates)} papers")

    llm_temperature = state.get("llm_temperature", 0)
    llm = get_llm(temperature=llm_temperature)

    logging.info(f"LLM scoring {len(papers)} survivors...")

    scored_papers = []
    for paper in papers:
        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator.
            Score how relevant a paper is to the user's query on a scale from 1 to 100.

            Consider:
            - Direct relevance to the query topic
            - Quality and depth of content based on the abstract
            - Potential usefulness for answering the query

            Respond with ONLY a number between 1 and 100, nothing else."""),
            HumanMessage(content=f"""User Query: {original_query}

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}
Abstract: {paper['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]

        response = llm.invoke(score_messages)
        paper['relevance_score'] = max(1, min(100, int(response.content.strip())))
        paper['relevance_score'] = final_relevance_score(paper)
        paper['source'] = 'arxiv'
        scored_papers.append(paper)
        logging.info(f"  📄 {paper['title'][:60]}... - Score: {paper['relevance_score']}")

    # Return OutputState fields
    return {
        "papers": scored_papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"Found {len(scored_papers)} papers on ArXiv for query: {query}",
                name="ArXivResearcher"
            )
        ]
    }


def wikipedia_researcher_node(state: InternalState) -> OutputState:
    """Wikipedia researcher node: searches Wikipedia and scores relevance."""

    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"Searching Wikipedia: '{query}' (iteration {iteration})")
    result = search_wikipedia.invoke({"topic": query, "sentences": 5})

    papers = state.get("papers", [])

    if result.get("success"):
        logging.info(f"Found Wikipedia article: {result['title']}")

        llm_temperature = state.get("llm_temperature", 0)

        # Score the article's relevance using LLM
        llm = get_llm(temperature=llm_temperature)

        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator.
            Score how relevant this Wikipedia article is to the user's query on a scale from 1 to 100.

            Consider:
            - Direct relevance to the query topic
            - Quality and depth of content based on the summary
            - Potential usefulness for answering the query

            Respond with ONLY a number between 1 and 100, nothing else."""),
            HumanMessage(content=f"""User Query: {original_query}

Article Title: {result['title']}
Summary: {result['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]

        response = llm.invoke(score_messages)
        relevance_score = int(response.content.strip())

        # Format Wikipedia result to match paper structure
        wiki_entry = {
            "id": result["url"],
            "title": result["title"],
            "summary": result["summary"],
            "url": result["url"],
            "relevance_score": max(1, min(100, relevance_score)),
            "source": "wikipedia",
            "authors": ["Wikipedia"],
            "published": "N/A"
        }

        papers.append(wiki_entry)
        logging.info(f"  📖 {result['title'][:60]}... - Score: {relevance_score}")

        message_content = f"Found Wikipedia article: {result['title']}"
    else:
        logging.warning(f"Wikipedia search failed: {result.get('error', 'Unknown error')}")
        message_content = f"Wikipedia search failed: {result.get('error', 'No results')}"

    # Return OutputState fields
    return {
        "papers": papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=message_content,
                name="WikipediaResearcher"
            )
        ]
    }


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


def summarizer_node(state: InternalState) -> OutputState:
    """Summarizer node: analyzes papers and generates a concise summary."""
    max_iterations = state.get("max_iterations", 2)
    llm_temperature = state.get("llm_temperature", 0)

    llm = get_llm(temperature=llm_temperature)

    papers = state["papers"]
    query = state["query"]
    iteration = state["iteration"]

    if len(papers) < 3 and iteration < max_iterations:
        logging.info(f"⚠️  Only {len(papers)} papers found, will retry search...")
        return {
            **state, 
            "summary": "NEED_MORE_PAPERS",
            "messages": [
                AIMessage(
                    content=f"Insufficient papers ({len(papers)}), retrying search...",
                    name="Summarizer"
                )
            ]
        }
    
    logging.info(f"📝 Synthesizing {len(papers)} papers...")
    
    papers_context = "\n\n".join([
        f"**Paper {i+1}:** {p['title']}\n"
        f"Authors: {', '.join(p['authors'][:3])}\n"
        f"Published: {p['published']}\n"
        f"Abstract: {p['summary'][:400]}...\n"
        f"URL: {p['url']}"
        for i, p in enumerate(papers)
    ])
    
    messages = [
        SystemMessage(content="""You are an expert scientific assistant. Respond in English only.
        Create a concise summary with:
        - 3-5 bullet points highlighting main insights of 3-5 sentences
        - Each bullet point must reference source papers [Paper X]
        - Clear and accessible style
        - End with a "References" section listing all papers

        Format:
        ## Summary
        • Point 1 [Paper 1, Paper 2]
        • Point 2 [Paper 3]
        ...

        ## References
        [Paper 1] Title - Authors (Year) - URL
        ..."""),
        HumanMessage(content=f"""Original question: {query}
        
Papers found:
{papers_context}

Generate a structured summary.""", name="User")
    ]
    
    response = llm.invoke(messages)
    summary = response.content
    
    logging.info("✅ Summary generated")
    
    # Return OutputState fields
    return {
        "summary": summary,
        "messages": [
            AIMessage(content=summary, name="Summarizer")
        ]
    }


def should_continue(state: InternalState) -> str:
    """
    Conditional edge: decides whether to loop back or end.

    This function is called after the summarizer node to determine
    the next step in the graph.

    Returns:
        - "researcher": if we need more papers (loops back)
        - "end": if we're done (exits the graph)
    """
    max_iterations = state.get("max_iterations", 2)

    if state.get("summary") == "NEED_MORE_PAPERS" and state["iteration"] < max_iterations:
        return "researcher"
    return "end"


# ============================================================================
# ASYNC STREAMING NODES
# ============================================================================

async def arxiv_researcher_node_streaming(state: InternalState) -> OutputState:
    """
    Async ArXiv researcher node with streaming support.

    This version uses the streaming ArXiv tool and streams token-level
    updates during relevance scoring.
    """
    max_papers = state.get("max_papers", 5)
    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"🔍 Searching ArXiv: '{query}' (iteration {iteration})")

    # Use streaming version of search
    papers = await search_arxiv_streaming.ainvoke({"query": query, "max_results": max_papers})
    logging.info(f"Found {len(papers)} papers")

    llm_temperature = state.get("llm_temperature", 0)

    # Score each paper's relevance using LLM with streaming
    llm = get_llm(temperature=llm_temperature)

    logging.info(f"Scoring paper relevance...")

    scored_papers = []
    for i, paper in enumerate(papers, 1):
        # Create prompt for scoring
        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator with expertise in academic paper assessment.

Score how relevant this paper is to the user's query on a scale from 1 to 100.

SCORING CRITERIA:

**High Relevance (80-100):**
- Directly addresses the core topic/question in the query
- Abstract demonstrates deep technical/conceptual alignment
- Contains specific methods, results, or insights that answer the query
- Published in reputable venue with clear contributions

**Moderate Relevance (50-79):**
- Partially addresses the query or covers related subtopics
- Abstract shows tangential connection or addresses one aspect of the query
- Provides useful background or context but not a direct answer
- May require inference to connect to the query

**Low Relevance (20-49):**
- Mentions query keywords but focuses on different aspects
- Abstract shows weak conceptual overlap
- Provides minimal useful information for the query
- May be from a related field but different application domain

**Irrelevant (1-19):**
- Shares only superficial keyword overlap
- Abstract shows the paper addresses a different problem entirely
- Would not help answer the query in any meaningful way

EVALUATION FACTORS:
1. **Topic alignment**: Does the paper's main focus match the query intent?
2. **Methodological relevance**: Are the techniques/approaches applicable to the query?
3. **Depth of coverage**: Does the abstract suggest comprehensive treatment of relevant concepts?
4. **Recency and impact**: Is the paper recent/seminal enough to provide current insights?
5. **Specificity**: Does it address the specific aspect mentioned in the query or just general concepts?

Respond with ONLY a number between 1 and 100. No explanation, no text, just the number."""),
            HumanMessage(content=f"""User Query: {original_query}

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}
Abstract: {paper['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]

        response = await llm.ainvoke(score_messages)
        relevance_score = int(response.content.strip())
        paper['relevance_score'] = max(1, min(100, relevance_score))
        paper['source'] = 'arxiv'
        scored_papers.append(paper)
        logging.info(f"  📄 {paper['title'][:60]}... - Score: {relevance_score}")

    # Return OutputState fields
    return {
        "papers": scored_papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=f"Found {len(scored_papers)} papers on ArXiv for query: {query}",
                name="ArXivResearcher"
            )
        ]
    }


async def wikipedia_researcher_node_streaming(state: InternalState) -> OutputState:
    """
    Async Wikipedia researcher node with streaming support.

    This version uses the streaming Wikipedia tool.
    """
    query = state["refined_query"]
    original_query = state["query"]
    iteration = state.get("iteration", 0)

    logging.info(f"🔍 Searching Wikipedia: '{query}' (iteration {iteration})")

    # Use streaming version of search
    result = await search_wikipedia_streaming.ainvoke({"topic": query, "sentences": 5})

    papers = state.get("papers", [])

    if result.get("success"):
        logging.info(f"Found Wikipedia article: {result['title']}")

        llm_temperature = state.get("llm_temperature", 0)

        # Score the article's relevance using LLM
        llm = get_llm(temperature=llm_temperature)

        score_messages = [
            SystemMessage(content="""You are a research relevance evaluator.
            Score how relevant this Wikipedia article is to the user's query on a scale from 1 to 100.

            Consider:
            - Direct relevance to the query topic
            - Quality and depth of content based on the summary
            - Potential usefulness for answering the query

            Respond with ONLY a number between 1 and 100, nothing else."""),
            HumanMessage(content=f"""User Query: {original_query}

Article Title: {result['title']}
Summary: {result['summary'][:500]}

Relevance Score (1-100):""", name="User")
        ]

        response = await llm.ainvoke(score_messages)
        relevance_score = int(response.content.strip())

        # Format Wikipedia result to match paper structure
        wiki_entry = {
            "id": result["url"],
            "title": result["title"],
            "summary": result["summary"],
            "url": result["url"],
            "relevance_score": max(1, min(100, relevance_score)),
            "source": "wikipedia",
            "authors": ["Wikipedia"],
            "published": "N/A"
        }

        papers.append(wiki_entry)
        logging.info(f"  📖 {result['title'][:60]}... - Score: {relevance_score}")

        message_content = f"Found Wikipedia article: {result['title']}"
    else:
        logging.warning(f"Wikipedia search failed: {result.get('error', 'Unknown error')}")
        message_content = f"Wikipedia search failed: {result.get('error', 'No results')}"

    # Return OutputState fields
    return {
        "papers": papers,
        "iteration": iteration + 1,
        "messages": [
            AIMessage(
                content=message_content,
                name="WikipediaResearcher"
            )
        ]
    }

async def summarizer_node_streaming(state: InternalState) -> OutputState:
    """
    Async summarizer node with token-level streaming.

    This version streams the summary generation token by token,
    providing real-time feedback to users.
    """
    max_iterations = state.get("max_iterations", 2)

    llm_temperature = state.get("llm_temperature", 0)

    llm = get_llm(temperature=llm_temperature)

    papers = state["papers"]
    query = state["query"]
    iteration = state["iteration"]

    if len(papers) < 3 and iteration < max_iterations:
        logging.info(f"⚠️  Only {len(papers)} papers found, will retry search...")
        return {
            **state,
            "summary": "NEED_MORE_PAPERS",
            "messages": [
                AIMessage(
                    content=f"Insufficient papers ({len(papers)}), retrying search...",
                    name="Summarizer"
                )
            ]
        }

    logging.info(f"📝 Synthesizing {len(papers)} papers with streaming...")

    papers_context = "\n\n".join([
        f"**Paper {i+1}:** {p['title']}\n"
        f"Authors: {', '.join(p['authors'][:3])}\n"
        f"Published: {p['published']}\n"
        f"Abstract: {p['summary'][:400]}...\n"
        f"URL: {p['url']}"
        for i, p in enumerate(papers)
    ])

    messages = [
        SystemMessage(content="""You are an expert scientific assistant.
        Create a concise summary with:
        - 3-5 bullet points highlighting main insights of 3-5 sentences
        - Each bullet point must reference source papers [Paper X]
        - Clear and accessible style
        - End with a "References" section listing all papers

        Format:
        ## Résumé
        • Point 1 [Paper 1, Paper 2]
        • Point 2 [Paper 3]
        ...

        ## Références
        [Paper 1] Title - Authors (Year) - URL
        ..."""),
        HumanMessage(content=f"""Original question: {query}

Papers found:
{papers_context}

Generate a structured summary.""", name="User")
    ]

    # Stream the response token by token
    summary_chunks = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            summary_chunks.append(chunk.content)

    summary = "".join(summary_chunks)

    logging.info("✅ Summary generated")

    # Return OutputState fields
    return {
        "summary": summary,
        "messages": [
            AIMessage(content=summary, name="Summarizer")
        ]
    }


def approver_node(state: InternalState) -> dict:
    """
    A node that explicitly requests human approval before proceeding.
    
    This demonstrates the node-level interrupt pattern where the node
    itself decides to pause and request input.
    """
    
    refined_query = state.get("refined_query", "")
    
    # This is where the magic happens - we interrupt and ask for a decision
    approval_response = interrupt({
        "type": "approval_request",
        "query": refined_query,
        "message": (
            f"The clarifier suggests this search query:\n\n"
            f"   '{refined_query}'\n\n"
            f"What would you like to do?\n"
            f"  - Type 'approve' to continue\n"
            f"  - Type 'edit' to modify the query\n"
            f"  - Type 'cancel' to stop"
        )
    })
    
    # Execution pauses at the interrupt() call above.
    # When you resume with Command(resume=...), that value appears here.
    
    action = approval_response.get("action", "cancel")
    
    if action == "cancel":
        return {
            "approved": False,
            "messages": [AIMessage(
                content="❌ Search cancelled by user",
                name="Approval"
            )]
        }
    
    if action == "edit":
        new_query = approval_response.get("new_query", refined_query)
        return {
            "refined_query": new_query,
            "approved": True,
            "messages": [AIMessage(
                content=f"✏️ Query updated to: {new_query}",
                name="Approval"
            )]
        }
    
    # If we get here, user approved
    return {
        "approved": True,
        "messages": [AIMessage(
            content="✅ Query approved - proceeding to search",
            name="Approval"
        )]
    }

def route_after_approval(state: InternalState) -> str:
    if state.get("approved", False):
        return "researcher"  # Approved - continue to search
    return "end"  # Cancelled - stop here


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

    llm = get_llm(temperature=state.get("llm_temperature", 0), max_tokens=8192)

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

    logging.info("Dual-audience summaries generated")

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


# ============================================================================
# PRE-HITL VALIDATION GATE
# ============================================================================

# Portable JSON Schema contracts derived from the Pydantic models. These can be
# exported/versioned independently of the producing code.
_CLINICIAN_SCHEMA = ClinicianSummary.model_json_schema()
_TECHNICAL_SCHEMA = TechnicalSummary.model_json_schema()


def validate_output_node(state: InternalState) -> dict:
    """Machine-validate the two draft summaries BEFORE a human ever sees them.

    Two layers:
      1. Structural — JSON Schema (required fields, types) per document.
      2. Grounding — every cited PMID must be one we actually retrieved
         (a cross-document constraint JSON Schema cannot express), so a
         hallucinated citation never reaches a clinical reviewer.

    On error it bumps `iteration` so a regenerate loop is bounded by
    `max_iterations`. The human then only ever adjudicates well-formed,
    grounded output — reviewing meaning, not form.
    """
    errors = []
    pairs = [
        ("clinician_summary", state.get("clinician_summary"), _CLINICIAN_SCHEMA),
        ("technical_summary", state.get("technical_summary"), _TECHNICAL_SCHEMA),
    ]

    # (1) structural — JSON Schema, intra-document
    for name, obj, schema in pairs:
        if obj is None:
            errors.append(f"{name}: missing")
            continue
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{name}: {exc.message}")

    # (2) grounding — citations must be a subset of retrieved PMIDs
    allowed = {p.get("pmid") for p in state.get("papers", []) if p.get("pmid")}
    if allowed:
        for name, obj, _ in pairs:
            if not obj:
                continue
            for ev in obj.get("evidence", []):
                pmid = ev.get("pmid")
                if pmid and pmid not in allowed:
                    errors.append(f"{name}: ungrounded citation pmid:{pmid}")

    if errors:
        logging.warning(f"⚠️  Pre-HITL validation found {len(errors)} issue(s): {errors}")
        return {"validation_errors": errors, "iteration": state.get("iteration", 0) + 1}
    return {"validation_errors": []}


def route_after_validation(state: InternalState) -> str:
    """Route to regenerate on validation errors, else to the human gate.

    Bounded by max_iterations so a persistently-malformed model cannot loop
    forever; after exhausting retries we let the human see it (they remain the
    final gate, and validation_errors are in state for display).
    """
    errors = state.get("validation_errors") or []
    if errors and state.get("iteration", 0) < state.get("max_iterations", 2):
        return "regenerate"
    return "approve"


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
        logging.info("Summaries approved by reviewer")
        return {
            "approved": True,
            "messages": [AIMessage(content="Summaries approved", name="HITL")]
        }

    logging.info("Summaries rejected by reviewer")
    return {
        "approved": False,
        "summary": "[REJECTED BY REVIEWER — summaries not finalized]",
        "messages": [AIMessage(content="Summaries rejected", name="HITL")]
    }


def route_after_hitl(state: InternalState) -> str:
    return "end"

