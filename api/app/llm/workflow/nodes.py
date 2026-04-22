"""Node creation functions for the Movie Night Assistant workflow.

This module contains factory functions that create LangGraph nodes.
Each node function processes the current state and returns state updates.
The factories accept agent dependencies and return closures that operate
on state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from app.llm.state import MAX_RETRIES, PASS_THRESHOLD, MovieNightState
from app.llm.workflow.formatters import (
    NO_MOVIES_FOUND_MESSAGE,
    RETRY_EXHAUSTED_FALLBACK_MESSAGE,
    format_candidate_list_response,
)
from app.schemas.domain import DraftRecommendation, EvaluationResult
from app.schemas.orchestrator import Constraints

if TYPE_CHECKING:
    from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
    from app.llm.evaluator_agent import EvaluatorAgent
    from app.llm.input_agent import InputOrchestratorAgent
    from app.llm.movie_finder_agent import MovieFinderAgent
    from app.llm.rag_agent import RAGAssistantAgent
    from app.llm.recommendation_agent import RecommendationWriterAgent
    from app.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


def create_orchestrate_node(
    orchestrator: OrchestratorAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the orchestrate node that classifies intent and extracts constraints.

    Note: This uses the simpler OrchestratorAgent that supports only movies/system
    routes. For full functionality (movies/rag/hybrid), use InputOrchestratorAgent
    with create_input_orchestrate_node instead.

    Args:
        orchestrator: The OrchestratorAgent instance.

    Returns:
        A node function that updates state with routing and constraints.
    """

    def orchestrate(state: MovieNightState) -> dict:
        user_message = state["user_message"]
        logger.info(f"Orchestrate node processing: {user_message[:50]}...")

        decision = orchestrator.decide(user_message)

        logger.debug(
            f"Orchestrate decision: route={decision.intent}, "
            f"needs_clarification={decision.needs_clarification}"
        )

        if decision.needs_clarification:
            clarification = (
                decision.clarification_question
                or "Could you please clarify what you're looking for?"
            )
            return {
                "route": "clarification",
                "constraints": decision.constraints,
                "final_response": clarification,
            }

        return {
            "route": decision.intent,
            "constraints": decision.constraints,
        }

    return orchestrate


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

        if decision.needs_clarification:
            clarification = (
                decision.clarification_question
                or "Could you please clarify what you're looking for?"
            )
            return {
                "route": "clarification",
                "constraints": decision.constraints,
                "needs_recommendation": False,
                "rag_query": None,
                "final_response": clarification,
            }

        return {
            "route": decision.route,
            "constraints": decision.constraints,
            "needs_recommendation": decision.needs_recommendation,
            "rag_query": decision.rag_query,
        }

    return input_orchestrate


def create_respond_node(
    movies_responder: MoviesResponder,
    system_responder: SystemResponder,
) -> Callable[[MovieNightState], dict]:
    """Create the respond node that generates the final response.

    For movie routes, this node uses the draft recommendation text when
    available (from RecommendationWriterAgent), or falls back to a
    simple formatted list of candidates.

    For RAG routes without dedicated RAG nodes, falls back to SystemResponder.

    Args:
        movies_responder: The MoviesResponder instance (fallback for simple responses).
        system_responder: The SystemResponder instance (fallback for RAG routes).

    Returns:
        A node function that generates the response based on route.
    """
    from app.llm.recommendation_agent import filter_candidates

    def respond(state: MovieNightState) -> dict:
        route = state.get("route")
        user_message = state["user_message"]
        constraints = state.get("constraints") or Constraints()
        candidate_movies = state.get("candidate_movies", [])
        rejected_titles = state.get("rejected_titles", [])
        draft: DraftRecommendation | None = state.get("draft_recommendation")
        evaluation_result: EvaluationResult | None = state.get("evaluation_result")
        retry_count = state.get("retry_count", 0)

        if route == "clarification":
            logger.info("Respond node: clarification already set, skipping")
            return {}

        logger.info(
            f"Respond node processing: route={route}, "
            f"candidates={len(candidate_movies)}, "
            f"rejected={len(rejected_titles)}, "
            f"has_draft={draft is not None}, "
            f"retry_count={retry_count}, "
            f"has_evaluation={evaluation_result is not None}"
        )

        if route in ("movies", "hybrid"):
            reply = _generate_movie_response(
                draft=draft,
                evaluation_result=evaluation_result,
                retry_count=retry_count,
                candidate_movies=candidate_movies,
                constraints=constraints,
                rejected_titles=rejected_titles,
            )
        elif route == "rag":
            reply = system_responder.respond(user_message)
        else:
            reply = system_responder.respond(user_message)

        return {"final_response": reply}

    def _generate_movie_response(
        draft: DraftRecommendation | None,
        evaluation_result: EvaluationResult | None,
        retry_count: int,
        candidate_movies: list,
        constraints: Constraints,
        rejected_titles: list[str],
    ) -> str:
        """Generate response for movie/hybrid routes."""
        if draft is not None:
            return draft.recommendation_text

        if evaluation_result is not None and retry_count >= MAX_RETRIES:
            logger.info(
                "Respond node: retries exhausted after evaluation failures; "
                "returning safe fallback"
            )
            return RETRY_EXHAUSTED_FALLBACK_MESSAGE

        safe_candidates = filter_candidates(
            candidate_movies, constraints, rejected_titles
        )
        if safe_candidates:
            return format_candidate_list_response(safe_candidates, constraints)

        return NO_MOVIES_FOUND_MESSAGE

    return respond


def create_find_movies_node(
    movie_finder: MovieFinderAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the find_movies node that retrieves candidate movies.

    This node uses the MovieFinderAgent to retrieve candidate movies
    based on user constraints. The candidates are stored in state for
    subsequent processing by the response node.

    Args:
        movie_finder: The MovieFinderAgent instance.

    Returns:
        A node function that populates candidate_movies in state.
    """

    def find_movies(state: MovieNightState) -> dict:
        constraints = state.get("constraints") or Constraints()
        rejected_titles = state.get("rejected_titles", [])

        logger.info(
            f"Find movies node: constraints={constraints}, "
            f"rejected={len(rejected_titles)} titles"
        )

        candidates = movie_finder.find_movies(
            constraints=constraints,
            limit=10,
            excluded_titles=rejected_titles,
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

    The draft is stored in state under ``draft_recommendation`` and is
    consumed by the respond node.

    Args:
        writer: The RecommendationWriterAgent instance.

    Returns:
        A node function that populates ``draft_recommendation`` in state.
    """

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
            return {"draft_recommendation": None}

        draft = writer.write(
            user_message=user_message,
            constraints=constraints,
            candidates=candidate_movies,
            rejected_titles=rejected_titles,
        )

        if draft is None:
            logger.info("Write recommendation node: writer returned None")
            return {"draft_recommendation": None}

        logger.info(
            f"Write recommendation node: drafted movie='{draft.movie.title}'"
        )
        return {"draft_recommendation": draft}

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

    If there is no draft to evaluate (e.g. the writer returned ``None``
    because no candidates survived filtering), the node returns no updates
    so ``respond`` can handle the empty case.

    Args:
        evaluator: The :class:`EvaluatorAgent` instance.

    Returns:
        A node function that updates ``evaluation_result``, and optionally
        ``retry_count``, ``rejected_titles`` and ``draft_recommendation``.
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
                "exhausted so the workflow proceeds to respond"
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
