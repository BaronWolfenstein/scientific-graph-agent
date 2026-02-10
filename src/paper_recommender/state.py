"""State definitions for paper recommender with long-term memory."""
from typing import TypedDict, List, Annotated
from typing import NotRequired
from operator import add
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# Pydantic schemas for Trustcall
class UserProfile(BaseModel):
    """User profile stored in long-term memory."""
    interests: List[str] = Field(description="Research topics of interest")
    read_papers: List[str] = Field(default=[], description="Paper IDs already read")
    expertise_level: str = Field(description="beginner, intermediate, or expert")


class PaperCollection(BaseModel):
    """Collection of papers organized by topic."""
    topic: str = Field(description="Research topic name")
    paper_ids: List[str] = Field(description="List of paper IDs for this topic")


# Graph state
class RecommenderState(TypedDict):
    """State for the paper recommender agent."""
    query: str
    user_profile: NotRequired[dict]
    collections: Annotated[List[dict], add]
    recommendations: NotRequired[List[dict]]
    messages: Annotated[List[BaseMessage], add]
    llm_model: NotRequired[str]
    max_recommendations: NotRequired[int]
