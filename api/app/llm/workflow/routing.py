"""Routing decision functions for the Movie Night Assistant workflow.

This module contains functions that determine workflow routing based on
current state. These are pure functions that examine state and return
the next node or END marker.
"""

import logging

from langgraph.graph import END

from app.llm.state import MAX_RETRIES, MovieNightState

logger = logging.getLogger(__name__)


def route_after_evaluate(state: MovieNightState) -> str:
    """Decide what happens after the evaluate node.

    Routes to:
    - ``respond`` if the draft is still present (it passed evaluation).
    - ``respond`` if there is no draft and no evaluation result (nothing
      was evaluated; e.g. no candidates survived filtering).
    - ``write_recommendation`` to retry if evaluation failed and we still
      have retries remaining.
    - ``respond`` otherwise (retries exhausted → safe fallback).

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    draft = state.get("draft_recommendation")
    evaluation_result = state.get("evaluation_result")
    retry_count = state.get("retry_count", 0) or 0

    if draft is not None:
        return "respond"

    if evaluation_result is None:
        return "respond"

    if retry_count < MAX_RETRIES:
        logger.info(
            f"Evaluate routing: retry {retry_count}/{MAX_RETRIES}; "
            "looping back to write_recommendation"
        )
        return "write_recommendation"

    logger.info(
        f"Evaluate routing: retries exhausted at {retry_count}; "
        "proceeding to respond with safe fallback"
    )
    return "respond"


def route_after_orchestrate(state: MovieNightState) -> str:
    """Determine the next node after orchestration (without RAG components).

    Routes to:
    - END if clarification is needed (response already set)
    - find_movies if route is movies or hybrid (need candidates)
    - respond for rag route or fallback

    Args:
        state: Current workflow state.

    Returns:
        Next node name or END.
    """
    route = state.get("route")

    if route == "clarification":
        return END

    if route in ("movies", "hybrid"):
        return "find_movies"

    return "respond"


def route_after_orchestrate_with_rag(state: MovieNightState) -> str:
    """Determine the next node after orchestration (RAG-enabled version).

    Routes to:
    - END if clarification is needed (response already set)
    - find_movies if route is movies (need candidates only)
    - find_movies if route is hybrid (need candidates, then RAG context)
    - rag_retrieve if route is rag (need RAG context, no movies)
    - respond as fallback

    Args:
        state: Current workflow state.

    Returns:
        Next node name or END.
    """
    route = state.get("route")

    if route == "clarification":
        return END

    if route in ("movies", "hybrid"):
        return "find_movies"

    if route == "rag":
        return "rag_retrieve"

    return "respond"


def route_after_find_movies_for_hybrid(state: MovieNightState) -> str:
    """Route after find_movies node for hybrid requests.

    For hybrid routes, retrieves RAG context before writing recommendations.
    For movies routes, goes directly to write_recommendation.

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    route = state.get("route")
    if route == "hybrid":
        return "rag_retrieve_hybrid"
    return "write_recommendation"


def should_respond(state: MovieNightState) -> str:
    """Determine if we should proceed to the respond node.

    If clarification was needed, skip the respond node since the
    final_response is already set.

    Args:
        state: Current workflow state.

    Returns:
        "respond" to continue to respond node, or END to finish.
    """
    if state.get("route") == "clarification":
        return END
    return "respond"
