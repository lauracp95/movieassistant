"""LangGraph workflow skeleton for the Movie Night Assistant.

This module provides the minimal executable graph that routes user
messages through the existing MVP behavior. The workflow wraps the
current orchestrator/responder flow while establishing the architectural
backbone for future phases.

Graph Shape (Phase 4 - Recommendation Writer Integration):

    START
      |
      v
    [input_orchestrate]  -- Classifies route (movies/rag/hybrid), extracts constraints
      |
      v (conditional)
      ├── clarification → END (response already set)
      ├── movies → [find_movies] → [write_recommendation] → [respond]
      ├── rag → [respond]
      └── hybrid → [find_movies] → [write_recommendation] → [respond]
      |
      v
    END

The write_recommendation node is inserted only when a
RecommendationWriterAgent is provided to the workflow. When absent,
the graph falls back to the Phase 3 behavior of formatting raw
candidates directly in the respond node.

Future phases will expand this to include:
- EvaluatorAgent node with retry loops
- RAGAssistantAgent node
"""

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import MovieFinderAgent
from app.llm.recommendation_agent import (
    RecommendationWriterAgent,
    filter_candidates,
)
from app.llm.state import MovieNightState
from app.schemas.domain import DraftRecommendation
from app.schemas.orchestrator import Constraints

logger = logging.getLogger(__name__)


def create_orchestrate_node(
    orchestrator: OrchestratorAgent,
) -> Callable[[MovieNightState], dict]:
    """Create the orchestrate node that classifies intent and extracts constraints.

    This is the legacy Phase 1 orchestrate node. Kept for backward compatibility.

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
    """Create the input orchestrate node for Phase 2 routing.

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

    For movie routes, this node uses candidate_movies from state to
    generate a response. If candidates are available, it formats them
    into a simple response. The RecommendationWriterAgent (future phase)
    will handle sophisticated response generation.

    Args:
        movies_responder: The MoviesResponder instance.
        system_responder: The SystemResponder instance.

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

        if route == "clarification":
            logger.info("Respond node: clarification already set, skipping")
            return {}

        logger.info(
            f"Respond node processing: route={route}, "
            f"candidates={len(candidate_movies)}, "
            f"rejected={len(rejected_titles)}, "
            f"has_draft={draft is not None}"
        )

        if route in ("movies", "hybrid"):
            if draft is not None:
                reply = draft.recommendation_text
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
    """Format candidate movies into a simple response.

    This is a temporary formatting function. The RecommendationWriterAgent
    (future phase) will replace this with sophisticated recommendation text.

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


def route_after_orchestrate(state: MovieNightState) -> str:
    """Determine the next node after orchestration.

    Routes to:
    - END if clarification is needed (response already set)
    - find_movies if route is movies or hybrid (need candidates)
    - respond if route is rag (no movie retrieval needed)

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

    Supports two modes:
    - Legacy mode: Uses OrchestratorAgent (Phase 1)
    - Input agent mode: Uses InputOrchestratorAgent (Phase 2/3)

    Phase 3 adds MovieFinderAgent for candidate retrieval.
    """

    def __init__(
        self,
        orchestrator: OrchestratorAgent | None,
        movies_responder: MoviesResponder,
        system_responder: SystemResponder,
        input_agent: InputOrchestratorAgent | None = None,
        movie_finder: MovieFinderAgent | None = None,
        recommendation_writer: RecommendationWriterAgent | None = None,
    ) -> None:
        """Initialize the workflow with agent instances.

        Args:
            orchestrator: The OrchestratorAgent for intent classification (legacy).
            movies_responder: The MoviesResponder for movie requests.
            system_responder: The SystemResponder for system questions.
            input_agent: The InputOrchestratorAgent for Phase 2 routing.
                        If provided, it takes precedence over orchestrator.
            movie_finder: The MovieFinderAgent for Phase 3 candidate retrieval.
                         If not provided, skips candidate retrieval step.
            recommendation_writer: The RecommendationWriterAgent for Phase 4
                composition. If provided, inserts a write_recommendation
                step between find_movies and respond.
        """
        self._orchestrator = orchestrator
        self._input_agent = input_agent
        self._movies_responder = movies_responder
        self._system_responder = system_responder
        self._movie_finder = movie_finder
        self._recommendation_writer = recommendation_writer
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the workflow graph.

        Uses InputOrchestratorAgent if available, otherwise falls back
        to legacy OrchestratorAgent. If MovieFinderAgent is provided,
        adds candidate retrieval before response generation.

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

        if self._movie_finder is not None:
            find_movies_node = create_find_movies_node(self._movie_finder)
            builder.add_node("find_movies", find_movies_node)

            builder.add_edge(START, "orchestrate")
            builder.add_conditional_edges("orchestrate", route_after_orchestrate)

            if self._recommendation_writer is not None:
                write_node = create_write_recommendation_node(
                    self._recommendation_writer
                )
                builder.add_node("write_recommendation", write_node)
                builder.add_edge("find_movies", "write_recommendation")
                builder.add_edge("write_recommendation", "respond")
            else:
                builder.add_edge("find_movies", "respond")

            builder.add_edge("respond", END)
        else:
            builder.add_edge(START, "orchestrate")
            builder.add_conditional_edges("orchestrate", should_respond)
            builder.add_edge("respond", END)

        return builder.compile()

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
