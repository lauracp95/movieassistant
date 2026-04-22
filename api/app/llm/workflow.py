"""LangGraph workflow for the Movie Night Assistant.

This module implements the complete recommendation workflow using LangGraph's
StateGraph. The workflow orchestrates multiple agents through a series of
nodes with conditional routing based on user intent.

Graph Shape:

    START
      │
      ▼
    [input_orchestrate]  ─ Classifies route (movies/rag/hybrid), extracts constraints
      │
      ▼ (conditional)
      ├── clarification → END (response already set)
      ├── rag → [rag_retrieve] → [rag_respond] → END
      ├── hybrid → [find_movies] → [rag_retrieve] → [write_recommendation] → [evaluate] → [respond] → END
      └── movies → [find_movies] → [write_recommendation] → [evaluate] → [respond] → END

Key Features:

- **Route Classification**: InputOrchestratorAgent determines whether the request
  needs movie recommendations (movies), knowledge retrieval (rag), or both (hybrid).

- **Movie Discovery**: MovieFinderAgent retrieves candidates from TMDB or stub data.

- **Recommendation Writing**: RecommendationWriterAgent generates grounded prose
  based on movie metadata and user constraints.

- **Quality Evaluation**: EvaluatorAgent validates recommendations against
  constraints and quality criteria, with automatic retry on failure.

- **RAG Integration**: DocumentRetriever and RAGAssistantAgent answer system
  questions using the knowledge base.

The workflow supports graceful degradation when optional agents are not provided.
"""

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.evaluator_agent import EvaluatorAgent
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import MovieFinderAgent
from app.llm.rag_agent import RAGAssistantAgent
from app.llm.recommendation_agent import (
    RecommendationWriterAgent,
    filter_candidates,
)
from app.llm.state import MAX_RETRIES, PASS_THRESHOLD, MovieNightState
from app.rag.retriever import DocumentRetriever
from app.schemas.domain import DraftRecommendation, EvaluationResult
from app.schemas.orchestrator import Constraints

RETRY_EXHAUSTED_FALLBACK_MESSAGE = (
    "I tried a few options but couldn't land on a recommendation that met the "
    "quality bar for your request. Could you rephrase or loosen your "
    "preferences a little?"
)

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
            if draft is not None:
                reply = draft.recommendation_text
            elif (
                evaluation_result is not None
                and retry_count >= MAX_RETRIES
            ):
                logger.info(
                    "Respond node: retries exhausted after evaluation failures; "
                    "returning safe fallback"
                )
                reply = RETRY_EXHAUSTED_FALLBACK_MESSAGE
            else:
                safe_candidates = filter_candidates(
                    candidate_movies, constraints, rejected_titles
                )
                if safe_candidates:
                    reply = _format_candidate_response(safe_candidates, constraints)
                else:
                    reply = (
                        "I couldn't find any movies matching your criteria. "
                        "Try broadening your search or specifying different preferences."
                    )
        elif route == "rag":
            reply = system_responder.respond(user_message)
        else:
            reply = system_responder.respond(user_message)

        return {"final_response": reply}

    return respond


def _format_candidate_response(
    candidates: list,
    constraints: Constraints,
) -> str:
    """Format candidate movies into a simple list response.

    This is a fallback formatter used when no draft recommendation is available.
    The primary path uses RecommendationWriterAgent for richer, grounded prose.

    Args:
        candidates: List of MovieResult objects.
        constraints: User constraints for context.

    Returns:
        Formatted response string.
    """
    if not candidates:
        return "I couldn't find any movies matching your criteria."

    lines = ["Here are some movie recommendations for you:\n"]

    for i, movie in enumerate(candidates[:5], 1):
        year_str = f" ({movie.year})" if movie.year else ""
        genres_str = ", ".join(movie.genres[:3]) if movie.genres else "Unknown genre"
        rating_str = f" - Rating: {movie.rating:.1f}/10" if movie.rating else ""
        runtime_str = f" - {movie.runtime_minutes} min" if movie.runtime_minutes else ""

        lines.append(f"{i}. **{movie.title}**{year_str}")
        lines.append(f"   {genres_str}{runtime_str}{rating_str}")

        if movie.overview:
            overview = movie.overview[:150] + "..." if len(movie.overview) > 150 else movie.overview
            lines.append(f"   {overview}")
        lines.append("")

    return "\n".join(lines)


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


class MovieNightWorkflow:
    """Wrapper class for the Movie Night Assistant LangGraph workflow.

    Encapsulates the graph construction and provides a simple interface
    for executing the workflow with a user message.

    The workflow can be configured with different combinations of agents:

    - **Full mode** (recommended): InputOrchestratorAgent + MovieFinderAgent +
      RecommendationWriterAgent + EvaluatorAgent + RAG components

    - **Minimal mode**: OrchestratorAgent only (limited to movies/system routes)

    All optional agents degrade gracefully when not provided.
    """

    def __init__(
        self,
        orchestrator: OrchestratorAgent | None,
        movies_responder: MoviesResponder,
        system_responder: SystemResponder,
        input_agent: InputOrchestratorAgent | None = None,
        movie_finder: MovieFinderAgent | None = None,
        recommendation_writer: RecommendationWriterAgent | None = None,
        evaluator: EvaluatorAgent | None = None,
        rag_retriever: DocumentRetriever | None = None,
        rag_agent: RAGAssistantAgent | None = None,
    ) -> None:
        """Initialize the workflow with agent instances.

        Args:
            orchestrator: The OrchestratorAgent for simple intent classification.
                Used only if input_agent is not provided (limited functionality).
            movies_responder: The MoviesResponder for fallback movie responses.
            system_responder: The SystemResponder for fallback system questions.
            input_agent: The InputOrchestratorAgent for full route classification
                (movies/rag/hybrid). Recommended - takes precedence over orchestrator.
            movie_finder: The MovieFinderAgent for candidate retrieval from TMDB.
                If not provided, no movie candidates are retrieved.
            recommendation_writer: The RecommendationWriterAgent for grounded prose.
                If provided, generates rich recommendation text instead of lists.
            evaluator: The EvaluatorAgent for draft validation with retry loop.
                Requires recommendation_writer. Validates constraints and quality.
            rag_retriever: The DocumentRetriever for knowledge base retrieval.
                Enables RAG-based responses for system questions.
            rag_agent: The RAGAssistantAgent for grounded answers from docs.
                Must be provided with rag_retriever for RAG functionality.
        """
        self._orchestrator = orchestrator
        self._input_agent = input_agent
        self._movies_responder = movies_responder
        self._system_responder = system_responder
        self._movie_finder = movie_finder
        self._recommendation_writer = recommendation_writer
        self._evaluator = evaluator
        self._rag_retriever = rag_retriever
        self._rag_agent = rag_agent
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the workflow graph.

        Uses InputOrchestratorAgent if available, otherwise falls back
        to basic OrchestratorAgent. If MovieFinderAgent is provided,
        adds candidate retrieval before response generation.

        RAG retrieval and response nodes are added for system questions
        and hybrid routes when rag_retriever and rag_agent are provided.

        Returns:
            Compiled StateGraph ready for execution.
        """
        builder = StateGraph(MovieNightState)

        if self._input_agent is not None:
            orchestrate_node = create_input_orchestrate_node(self._input_agent)
        elif self._orchestrator is not None:
            orchestrate_node = create_orchestrate_node(self._orchestrator)
        else:
            raise ValueError("Either orchestrator or input_agent must be provided")

        respond_node = create_respond_node(
            self._movies_responder, self._system_responder
        )

        builder.add_node("orchestrate", orchestrate_node)
        builder.add_node("respond", respond_node)

        has_rag = self._rag_retriever is not None and self._rag_agent is not None

        if has_rag:
            rag_retrieve_node = create_rag_retrieve_node(self._rag_retriever)
            rag_respond_node = create_rag_respond_node(self._rag_agent)
            builder.add_node("rag_retrieve", rag_retrieve_node)
            builder.add_node("rag_respond", rag_respond_node)

        if self._movie_finder is not None:
            find_movies_node = create_find_movies_node(self._movie_finder)
            builder.add_node("find_movies", find_movies_node)

            builder.add_edge(START, "orchestrate")

            if has_rag:
                builder.add_conditional_edges(
                    "orchestrate",
                    route_after_orchestrate_with_rag,
                    {
                        END: END,
                        "find_movies": "find_movies",
                        "rag_retrieve": "rag_retrieve",
                        "respond": "respond",
                    },
                )
                builder.add_edge("rag_retrieve", "rag_respond")
                builder.add_edge("rag_respond", END)
            else:
                builder.add_conditional_edges("orchestrate", route_after_orchestrate)

            if self._recommendation_writer is not None:
                write_node = create_write_recommendation_node(
                    self._recommendation_writer
                )
                builder.add_node("write_recommendation", write_node)

                if has_rag:
                    builder.add_conditional_edges(
                        "find_movies",
                        self._route_after_find_movies,
                        {
                            "rag_retrieve_hybrid": "rag_retrieve_hybrid",
                            "write_recommendation": "write_recommendation",
                        },
                    )
                    rag_retrieve_hybrid_node = create_rag_retrieve_node(
                        self._rag_retriever
                    )
                    builder.add_node("rag_retrieve_hybrid", rag_retrieve_hybrid_node)
                    builder.add_edge("rag_retrieve_hybrid", "write_recommendation")
                else:
                    builder.add_edge("find_movies", "write_recommendation")

                if self._evaluator is not None:
                    evaluate_node = create_evaluate_node(self._evaluator)
                    builder.add_node("evaluate", evaluate_node)
                    builder.add_edge("write_recommendation", "evaluate")
                    builder.add_conditional_edges(
                        "evaluate",
                        route_after_evaluate,
                        {
                            "respond": "respond",
                            "write_recommendation": "write_recommendation",
                        },
                    )
                else:
                    builder.add_edge("write_recommendation", "respond")
            else:
                builder.add_edge("find_movies", "respond")

            builder.add_edge("respond", END)
        else:
            builder.add_edge(START, "orchestrate")
            if has_rag:
                builder.add_conditional_edges(
                    "orchestrate",
                    route_after_orchestrate_with_rag,
                    {
                        END: END,
                        "rag_retrieve": "rag_retrieve",
                        "respond": "respond",
                    },
                )
                builder.add_edge("rag_retrieve", "rag_respond")
                builder.add_edge("rag_respond", END)
            else:
                builder.add_conditional_edges("orchestrate", should_respond)
            builder.add_edge("respond", END)

        return builder.compile()

    def _route_after_find_movies(self, state: MovieNightState) -> str:
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

    def invoke(self, user_message: str) -> MovieNightState:
        """Execute the workflow with a user message.

        Args:
            user_message: The user's input message.

        Returns:
            The final workflow state containing the response.
        """
        initial_state: MovieNightState = {
            "user_message": user_message,
            "route": None,
            "constraints": None,
            "needs_recommendation": False,
            "rag_query": None,
            "candidate_movies": [],
            "retrieved_contexts": [],
            "draft_recommendation": None,
            "evaluation_result": None,
            "retry_count": 0,
            "rejected_titles": [],
            "final_response": None,
            "error": None,
        }

        logger.info(f"Workflow invoked with message: {user_message[:50]}...")
        result = self._graph.invoke(initial_state)
        logger.info("Workflow completed")

        return result

    def get_response(self, user_message: str) -> tuple[str, str | None, Constraints | None]:
        """Execute the workflow and extract the response details.

        Convenience method that runs the workflow and extracts the
        commonly needed response fields.

        Args:
            user_message: The user's input message.

        Returns:
            Tuple of (reply, route, constraints).

        Raises:
            RuntimeError: If the workflow fails to produce a response.
        """
        result = self.invoke(user_message)

        final_response = result.get("final_response")
        if not final_response:
            raise RuntimeError("Workflow did not produce a response")

        return (
            final_response,
            result.get("route"),
            result.get("constraints"),
        )
