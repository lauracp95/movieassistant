"""Shared state for the Movie Night Assistant LangGraph workflow.

This module defines the MovieNightState TypedDict used by the LangGraph
workflow to maintain state across nodes. It also defines configuration
constants for the workflow behavior.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.schemas.domain import (
    DraftRecommendation,
    EvaluationResult,
    MovieResult,
    RetrievedContext,
)
from app.schemas.orchestrator import Constraints

MAX_RETRIES: int = 3
PASS_THRESHOLD: float = 0.7
MAX_MOVIE_SEARCHES: int = 5

RouteType = Literal["movies", "rag", "hybrid", "clarification"]


class MovieNightState(TypedDict, total=False):
    """State object for the Movie Night Assistant workflow.

    This TypedDict defines all fields that can be passed between nodes
    in the LangGraph workflow. Fields are marked as optional (total=False)
    to allow incremental state building.

    Attributes:
        messages: Conversation history with automatic message accumulation.
        user_message: The current user message being processed.
        route: The determined processing route (movies, rag, hybrid, or clarification).
        constraints: Extracted constraints from the user message.
        needs_recommendation: Whether a movie recommendation should be generated.
        rag_query: Query for RAG pipeline when route is rag or hybrid.
        candidate_movies: Movies retrieved from external sources.
        retrieved_contexts: RAG or external context for augmentation.
        draft_recommendation: Current recommendation being evaluated.
        evaluation_result: Result of the last evaluation.
        retry_count: Number of retry attempts for the current request.
        rejected_titles: Movie titles that were rejected during evaluation.
        final_response: The final response to return to the user.
        error: Error message if something went wrong.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    route: RouteType | None
    constraints: Constraints | None
    needs_recommendation: bool
    rag_query: str | None
    candidate_movies: list[MovieResult]
    retrieved_contexts: list[RetrievedContext]
    draft_recommendation: DraftRecommendation | None
    evaluation_result: EvaluationResult | None
    retry_count: int
    rejected_titles: list[str]
    final_response: str | None
    error: str | None


def create_initial_state(user_message: str) -> MovieNightState:
    """Create an initial state for a new workflow invocation.

    Args:
        user_message: The user's input message.

    Returns:
        A MovieNightState with default values and the user message set.
    """
    return MovieNightState(
        messages=[],
        user_message=user_message,
        route=None,
        constraints=None,
        needs_recommendation=False,
        rag_query=None,
        candidate_movies=[],
        retrieved_contexts=[],
        draft_recommendation=None,
        evaluation_result=None,
        retry_count=0,
        rejected_titles=[],
        final_response=None,
        error=None,
    )
