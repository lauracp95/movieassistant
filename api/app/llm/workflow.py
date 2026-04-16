"""LangGraph workflow skeleton for the Movie Night Assistant.

This module provides the minimal executable graph that routes user
messages through the existing MVP behavior. The workflow wraps the
current orchestrator/responder flow while establishing the architectural
backbone for future phases.

Graph Shape (Phase 2 - Input Orchestration):

    START
      |
      v
    [input_orchestrate]  -- Classifies route (movies/rag/hybrid), extracts constraints
      |
      v (conditional)
      ├── clarification → END (response already set)
      ├── movies → [respond]
      ├── rag → [respond]
      └── hybrid → [respond]
      |
      v
    END

Future phases will expand this to include:
- MovieFinderAgent node
- RecommendationWriterAgent node
- EvaluatorAgent node with retry loops
- RAGAssistantAgent node
"""

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.state import MovieNightState
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

        if route == "clarification":
            logger.info("Respond node: clarification already set, skipping")
            return {}

        logger.info(f"Respond node processing: route={route}")

        if route == "movies":
            reply = movies_responder.respond(user_message, constraints)
        elif route == "rag":
            reply = system_responder.respond(user_message)
        elif route == "hybrid":
            reply = movies_responder.respond(user_message, constraints)
        else:
            reply = system_responder.respond(user_message)

        return {"final_response": reply}

    return respond


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
    - Input agent mode: Uses InputOrchestratorAgent (Phase 2)
    """

    def __init__(
        self,
        orchestrator: OrchestratorAgent | None,
        movies_responder: MoviesResponder,
        system_responder: SystemResponder,
        input_agent: InputOrchestratorAgent | None = None,
    ) -> None:
        """Initialize the workflow with agent instances.

        Args:
            orchestrator: The OrchestratorAgent for intent classification (legacy).
            movies_responder: The MoviesResponder for movie requests.
            system_responder: The SystemResponder for system questions.
            input_agent: The InputOrchestratorAgent for Phase 2 routing.
                        If provided, it takes precedence over orchestrator.
        """
        self._orchestrator = orchestrator
        self._input_agent = input_agent
        self._movies_responder = movies_responder
        self._system_responder = system_responder
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the workflow graph.

        Uses InputOrchestratorAgent if available, otherwise falls back
        to legacy OrchestratorAgent.

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
