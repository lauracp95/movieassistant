"""Routing decision functions for the Movie Night Assistant workflow.

This module contains functions that determine workflow routing based on
current state. These are pure functions that examine state and return
the next node or END marker.
"""

import logging

from langgraph.graph import END

from app.workflow.state import MAX_RETRIES, MovieNightState

logger = logging.getLogger(__name__)


def route_after_evaluate(state: MovieNightState) -> str:
    """Decide what happens after the evaluate node.

    Routes to:
    - END if the draft is still present (it passed evaluation).
    - END if there is no draft and no evaluation result (nothing
      was evaluated; e.g. no candidates survived filtering).
    - ``write_recommendation`` to retry if evaluation failed and we still
      have retries remaining.
    - END otherwise (retries exhausted → safe fallback already set in state).

    Args:
        state: Current workflow state.

    Returns:
        Next node name or END.
    """
    draft = state.get("draft_recommendation")
    evaluation_result = state.get("evaluation_result")
    retry_count = state.get("retry_count", 0) or 0

    if draft is not None:
        return END

    if evaluation_result is None:
        return END

    if retry_count < MAX_RETRIES:
        logger.info(
            f"Evaluate routing: retry {retry_count}/{MAX_RETRIES}; "
            "looping back to write_recommendation"
        )
        return "write_recommendation"

    logger.info(
        f"Evaluate routing: retries exhausted at {retry_count}; "
        "proceeding to END with safe fallback"
    )
    return END


def route_after_orchestrate(state: MovieNightState) -> str:
    """Determine the next node after orchestration.

    Routes to:
    - END if clarification is needed (response already set)
    - find_movies if route is movies or hybrid (need candidates)
    - rag_retrieve for rag route
    - END for any unknown route (clarification response set by the node)

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

    logger.warning("Unknown route '%s' after orchestration; routing to END", route)
    return END


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
