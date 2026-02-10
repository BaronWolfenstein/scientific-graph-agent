"""Paper recommender graph with long-term memory and listeners."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

from .state import RecommenderState
from .nodes import (
    profile_manager_node,
    recommender_node,
    collection_manager_node
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_memory_changes(state: RecommenderState):
    """
    Listener function to log memory changes.

    This function is called after each node execution to track
    changes to long-term memory (profile and collections).
    """
    profile = state.get("user_profile", {})
    collections = state.get("collections", [])

    logger.info("=== MEMORY STATE ===")
    logger.info(f"Profile interests: {profile.get('interests', [])}")
    logger.info(f"Profile expertise: {profile.get('expertise_level', 'beginner')}")
    logger.info(f"Collections: {len(collections)} topics")

    for coll in collections:
        topic = coll.get("topic", "Unknown")
        count = len(coll.get("paper_ids", []))
        logger.info(f"  - {topic}: {count} papers")

    logger.info("===================")


def create_recommender_graph(with_checkpointer: bool = True):
    """
    Creates paper recommender graph with long-term memory.

    The graph demonstrates:
    1. Profile-based memory: User preferences tracked with Trustcall
    2. Collection-based memory: Papers organized by topic with Trustcall
    3. Memory change tracking: Listeners log all memory updates

    Flow:
        START -> profile_manager -> recommender -> collection_manager -> END

    Args:
        with_checkpointer: Enable memory persistence

    Returns:
        Compiled graph with memory and listeners
    """
    workflow = StateGraph(RecommenderState)

    # Add nodes
    workflow.add_node("profile_manager", profile_manager_node)
    workflow.add_node("recommender", recommender_node)
    workflow.add_node("collection_manager", collection_manager_node)

    # Define edges
    workflow.set_entry_point("profile_manager")
    workflow.add_edge("profile_manager", "recommender")
    workflow.add_edge("recommender", "collection_manager")
    workflow.add_edge("collection_manager", END)

    logger.info("Paper recommender graph configured")

    if with_checkpointer:
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)

        # Compile with checkpointer
        # Note: with_listeners() is not yet supported by langgraph-api
        # You can add listeners when using the graph programmatically
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


# Default graph instance for LangGraph Studio
# Note: Listeners need to be added when invoking the graph programmatically
# graph.with_listeners(on_end=log_memory_changes)
graph = create_recommender_graph(with_checkpointer=False)
