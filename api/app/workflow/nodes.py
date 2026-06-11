"""Node creation functions for the Movie Night Assistant workflow.

This module contains factory functions that create LangGraph nodes.
Each node function processes the current state and returns state updates.
The factories accept agent dependencies and return closures that operate
on state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from app.workflow.state import MAX_RETRIES, PASS_THRESHOLD, MovieNightState
from app.workflow.formatters import (
    NO_MOVIES_FOUND_MESSAGE,
    RETRY_EXHAUSTED_FALLBACK_MESSAGE,
    format_candidate_list_response,
)
from app.schemas.domain import DraftRecommendation, EvaluationResult
from app.schemas.input import Constraints

if TYPE_CHECKING:
    from app.agents import EvaluatorAgent
    from app.agents import InputOrchestratorAgent
    from app.agents import RAGAssistantAgent
    from app.agents import RecommendationWriterAgent
    from app.agents import MovieFinderAgent
    from app.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


def create_input_orchestrate_node(
    input_agent: InputOrchestratorAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the input orchestrate node for full route classification.

    This node uses the InputOrchestratorAgent to classify routes as
    movies, rag, or hybrid, extract constraints, and generate RAG queries.

    Args:
        input_agent: The InputOrchestratorAgent instance.

    Returns:
        A node function that updates state with rich routing information.
    """

    def input_orchestrate(state: MovieNightState) -> dict:
        user_message = state["user_message"]
        logger.info(f"Input orchestrate node processing: {user_message[:50]}...")

        decision = input_agent.decide(user_message)

        logger.debug(
            f"Input decision: route={decision.route}, "
            f"needs_clarification={decision.needs_clarification}, "
            f"needs_recommendation={decision.needs_recommendation}"
        )

        if decision.search_query and not decision.search_query.is_empty():
            logger.info(
                f"Extracted search query: actors={decision.search_query.actors}, "
                f"directors={decision.search_query.directors}, "
                f"year={decision.search_query.year}, "
                f"year_range=({decision.search_query.year_start}, {decision.search_query.year_end}), "
                f"keywords={decision.search_query.keywords}"
            )

        if decision.needs_clarification:
            clarification = (
                decision.clarification_question
                or "Could you please clarify what you're looking for?"
            )
            return {
                "route": "clarification",
                "constraints": decision.constraints,
                "search_query": None,
                "needs_recommendation": False,
                "rag_query": None,
                "final_response": clarification,
            }

        _KNOWN_ROUTES = {"movies", "rag", "hybrid"}
        if decision.route not in _KNOWN_ROUTES:
            logger.warning(
                "Unrecognised route '%s' from orchestrator; defaulting to clarification",
                decision.route,
            )
            return {
                "route": "clarification",
                "constraints": decision.constraints,
                "search_query": None,
                "needs_recommendation": False,
                "rag_query": None,
                "final_response": "Could you please clarify what you're looking for?",
            }

        return {
            "route": decision.route,
            "constraints": decision.constraints,
            "search_query": decision.search_query,
            "needs_recommendation": decision.needs_recommendation,
            "rag_query": decision.rag_query,
        }

    return input_orchestrate


def create_find_movies_node(
    movie_finder: MovieFinderAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the find_movies node that retrieves candidate movies.

    This node uses the MovieFinderAgent to retrieve candidate movies
    based on user constraints and rich search query. The candidates are
    stored in state for subsequent processing by the response node.

    Args:
        movie_finder: The MovieFinderAgent instance.

    Returns:
        A node function that populates candidate_movies in state.
    """

    def find_movies(state: MovieNightState) -> dict:
        constraints = state.get("constraints") or Constraints()
        search_query = state.get("search_query")
        rejected_titles = state.get("rejected_titles", [])

        logger.info(
            f"Find movies node: constraints={constraints}, "
            f"search_query={search_query is not None}, "
            f"rejected={len(rejected_titles)} titles"
        )

        candidates = movie_finder.find_movies(
            constraints=constraints,
            limit=10,
            excluded_titles=rejected_titles,
            search_query=search_query,
        )

        logger.info(f"Find movies node found {len(candidates)} candidates")

        return {"candidate_movies": candidates}

    return find_movies


def create_write_recommendation_node(
    writer: RecommendationWriterAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the write_recommendation node.

    This node separates recommendation composition from candidate retrieval.
    It consumes ``candidate_movies``, ``constraints``, ``user_message`` and
    ``rejected_titles`` from state and produces a ``DraftRecommendation``.

    The draft and its text are stored in state under ``draft_recommendation``
    and ``final_response`` respectively.

    Args:
        writer: The RecommendationWriterAgent instance.

    Returns:
        A node function that populates ``draft_recommendation`` and
        ``final_response`` in state.
    """
    from app.workflow.candidate_selector import filter_candidates

    def write_recommendation(state: MovieNightState) -> dict:
        user_message = state.get("user_message", "")
        constraints = state.get("constraints") or Constraints()
        candidate_movies = state.get("candidate_movies", [])
        rejected_titles = state.get("rejected_titles", [])

        logger.info(
            "Write recommendation node: "
            f"candidates={len(candidate_movies)}, "
            f"rejected={len(rejected_titles)}"
        )

        if not candidate_movies:
            logger.info("Write recommendation node: no candidates, skipping")
            return {"draft_recommendation": None, "final_response": NO_MOVIES_FOUND_MESSAGE}

        draft = writer.write(
            user_message=user_message,
            constraints=constraints,
            candidates=candidate_movies,
            rejected_titles=rejected_titles,
        )

        if draft is None:
            logger.info("Write recommendation node: writer returned None")
            safe_candidates = filter_candidates(candidate_movies, constraints, rejected_titles)
            final_response = (
                format_candidate_list_response(safe_candidates, constraints)
                if safe_candidates
                else NO_MOVIES_FOUND_MESSAGE
            )
            return {"draft_recommendation": None, "final_response": final_response}

        logger.info(
            f"Write recommendation node: drafted movie='{draft.movie.title}'"
        )
        return {"draft_recommendation": draft, "final_response": draft.recommendation_text}

    return write_recommendation


def create_evaluate_node(
    evaluator: EvaluatorAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the evaluate node that validates draft recommendations.

    On each run, this node asks the :class:`EvaluatorAgent` to score the
    current ``draft_recommendation``. The evaluator's ``passed`` flag is
    combined with :data:`PASS_THRESHOLD` to determine whether the draft is
    accepted. On failure, the node updates state so the workflow can loop
    back into the writer with a different candidate:

    - ``retry_count`` is incremented
    - the failed ``draft_recommendation.movie.title`` is appended to
      ``rejected_titles``
    - ``draft_recommendation`` is cleared
    - when retries are exhausted, ``final_response`` is set to the safe
      fallback message so the workflow can proceed to END

    If there is no draft to evaluate (e.g. the writer returned ``None``
    because no candidates survived filtering), the node returns no updates
    so ``route_after_evaluate`` can proceed to END with ``final_response``
    already set by the write_recommendation node.

    Args:
        evaluator: The :class:`EvaluatorAgent` instance.

    Returns:
        A node function that updates ``evaluation_result``, and optionally
        ``retry_count``, ``rejected_titles``, ``draft_recommendation``,
        and ``final_response``.
    """

    def evaluate(state: MovieNightState) -> dict:
        draft: DraftRecommendation | None = state.get("draft_recommendation")
        constraints = state.get("constraints") or Constraints()
        rejected_titles = list(state.get("rejected_titles", []) or [])
        retry_count = state.get("retry_count", 0) or 0
        user_message = state.get("user_message", "")

        if draft is None:
            logger.info(
                "Evaluate node: no draft to evaluate; marking retries as "
                "exhausted so the workflow proceeds to END"
            )
            return {"retry_count": MAX_RETRIES}

        logger.info(
            f"Evaluate node: judging draft for '{draft.movie.title}' "
            f"(retry_count={retry_count}, rejected={len(rejected_titles)})"
        )

        result = evaluator.evaluate(
            user_message=user_message,
            constraints=constraints,
            draft=draft,
            rejected_titles=rejected_titles,
        )

        passed = result.passed and result.score >= PASS_THRESHOLD

        updates: dict = {"evaluation_result": result}

        if passed:
            logger.info(
                f"Evaluate node: draft for '{draft.movie.title}' PASSED "
                f"(score={result.score:.2f})"
            )
            return updates

        logger.info(
            f"Evaluate node: draft for '{draft.movie.title}' FAILED "
            f"(score={result.score:.2f}, passed={result.passed}); "
            f"incrementing retry_count and appending to rejected_titles"
        )

        if draft.movie.title not in rejected_titles:
            rejected_titles.append(draft.movie.title)

        updates["retry_count"] = retry_count + 1
        updates["rejected_titles"] = rejected_titles
        updates["draft_recommendation"] = None

        if updates["retry_count"] >= MAX_RETRIES:
            logger.info(
                f"Evaluate node: retries exhausted at {updates['retry_count']}; "
                "setting safe fallback response"
            )
            updates["final_response"] = RETRY_EXHAUSTED_FALLBACK_MESSAGE

        return updates

    return evaluate


def create_rag_retrieve_node(
    retriever: DocumentRetriever,
) -> Callable[[MovieNightState], dict]:
    """Create the rag_retrieve node that retrieves relevant documents.

    This node uses the DocumentRetriever to search the knowledge base
    for documents relevant to the user's RAG query. Results are stored
    in state under ``retrieved_contexts``.

    Args:
        retriever: The DocumentRetriever instance.

    Returns:
        A node function that populates ``retrieved_contexts`` in state.
    """

    def rag_retrieve(state: MovieNightState) -> dict:
        rag_query = state.get("rag_query")
        user_message = state.get("user_message", "")

        query = rag_query or user_message

        logger.info(f"RAG retrieve node: query='{query[:50]}...'")

        contexts = retriever.retrieve(query)

        logger.info(f"RAG retrieve node: found {len(contexts)} relevant contexts")

        return {"retrieved_contexts": contexts}

    return rag_retrieve


def create_rag_respond_node(
    rag_agent: RAGAssistantAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the rag_respond node that generates RAG-grounded answers.

    This node uses the RAGAssistantAgent to generate an answer based on
    retrieved contexts. It is used for pure RAG routes (system questions).

    Args:
        rag_agent: The RAGAssistantAgent instance.

    Returns:
        A node function that populates ``final_response`` in state.
    """

    def rag_respond(state: MovieNightState) -> dict:
        user_message = state.get("user_message", "")
        rag_query = state.get("rag_query")
        contexts = state.get("retrieved_contexts", [])

        query = rag_query or user_message

        logger.info(
            f"RAG respond node: query='{query[:50]}...', "
            f"contexts={len(contexts)}"
        )

        answer = rag_agent.answer(query=query, contexts=contexts)

        logger.info(f"RAG respond node: generated answer length={len(answer)}")

        return {"final_response": answer}

    return rag_respond
