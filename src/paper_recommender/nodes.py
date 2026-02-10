"""Nodes for paper recommender agent using Trustcall for memory management."""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from trustcall import create_extractor
import logging

from .state import RecommenderState, UserProfile, PaperCollection
from .tools import search_recent_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create Trustcall extractors using schemas from state.py
profile_extractor = create_extractor(
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[UserProfile],
    tool_choice="UserProfile"
)

collection_extractor = create_extractor(
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[PaperCollection],
    tool_choice="PaperCollection",
    enable_inserts=True
)


def profile_manager_node(state: RecommenderState) -> dict:
    """
    Updates user profile using Trustcall extraction.

    Uses profile-based memory to track user interests and expertise.
    """
    query = state["query"]
    current_profile = state.get("user_profile", {
        "interests": [],
        "read_papers": [],
        "expertise_level": "beginner"
    })

    logger.info("Extracting profile from query")

    prompt = f"""Based on this user query, extract their research interests and expertise level.

Query: {query}

Current interests: {current_profile.get('interests', [])}

Extract:
- interests: List of research topics mentioned
- expertise_level: beginner, intermediate, or expert
- read_papers: Keep existing list {current_profile.get('read_papers', [])}
"""

    result = profile_extractor.invoke({"messages": [{"role": "user", "content": prompt}]})

    # Extract profile from Trustcall result
    if result and "responses" in result and len(result["responses"]) > 0:
        extracted = result["responses"][0]

        # Merge with current profile
        current_interests = set(current_profile.get("interests", []))
        new_interests = set(extracted.interests)
        merged_interests = list(current_interests.union(new_interests))

        updated_profile = {
            "interests": merged_interests,
            "read_papers": extracted.read_papers,
            "expertise_level": extracted.expertise_level
        }

        logger.info(f"Profile updated: {len(merged_interests)} interests, level: {extracted.expertise_level}")
    else:
        updated_profile = current_profile
        logger.info("No profile updates")

    return {
        "user_profile": updated_profile,
        "messages": [AIMessage(
            content=f"Profile updated",
            name="ProfileManager"
        )]
    }


def recommender_node(state: RecommenderState) -> dict:
    """
    Generates paper recommendations based on user profile.

    Fetches recent papers relevant to user interests.
    """
    query = state["query"]
    profile = state.get("user_profile", {})
    interests = profile.get("interests", [])
    max_recs = state.get("max_recommendations", 5)

    logger.info("Generating recommendations")

    # Determine search topic
    if interests:
        search_topic = " ".join(interests[:2])
    else:
        search_topic = query

    # Fetch recent papers using tool
    papers = search_recent_papers.invoke({"topic": search_topic, "max_results": max_recs})

    logger.info(f"Found {len(papers)} papers")

    return {
        "recommendations": papers,
        "messages": [AIMessage(
            content=f"Found {len(papers)} papers",
            name="Recommender"
        )]
    }


def collection_manager_node(state: RecommenderState) -> dict:
    """
    Manages paper collections using Trustcall.

    Uses collection-based memory with enable_inserts=True to add new papers.
    """
    recommendations = state.get("recommendations", [])
    current_collections = state.get("collections", [])

    logger.info("Managing paper collections")

    # Format existing collections for Trustcall
    existing_data = [
        (str(i), "PaperCollection", coll)
        for i, coll in enumerate(current_collections)
    ]

    # Create prompt with new recommendations
    papers_text = "\n".join([
        f"- {p['title']} (ID: {p['id']})"
        for p in recommendations
    ])

    prompt = f"""Based on these new paper recommendations, organize them by topic.

Papers:
{papers_text}

Group papers by research topic. For each topic group, provide:
- topic: The research topic name
- paper_ids: List of paper IDs for this topic
"""

    result = collection_extractor.invoke({
        "messages": [{"role": "user", "content": prompt}],
        "existing": existing_data
    })

    # Extract updated collections
    updated_collections = []
    if result and "responses" in result:
        for response in result["responses"]:
            collection = {
                "topic": response.topic,
                "paper_ids": response.paper_ids
            }
            updated_collections.append(collection)

        logger.info(f"Updated collections: {len(updated_collections)} topics")
    else:
        updated_collections = current_collections
        logger.info("No collection updates")

    return {
        "collections": updated_collections,
        "messages": [AIMessage(
            content=f"Updated collections",
            name="CollectionManager"
        )]
    }
