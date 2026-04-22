"""LangGraph workflow builder for the Movie Night Assistant.

This module contains the MovieNightWorkflow class that constructs and
manages the LangGraph StateGraph for the recommendation workflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.llm.state import MovieNightState
from app.llm.workflow.nodes import (
    create_evaluate_node,
    create_find_movies_node,
    create_input_orchestrate_node,
    create_orchestrate_node,
    create_rag_respond_node,
    create_rag_retrieve_node,
    create_respond_node,
    create_write_recommendation_node,
)
from app.llm.workflow.routing import (
    route_after_evaluate,
    route_after_orchestrate,
    route_after_orchestrate_with_rag,
    route_after_find_movies_for_hybrid,
    should_respond,
)
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

        orchestrate_node = self._create_orchestrate_node()
        respond_node = create_respond_node(
            self._movies_responder, self._system_responder
        )

        builder.add_node("orchestrate", orchestrate_node)
        builder.add_node("respond", respond_node)

        has_rag = self._rag_retriever is not None and self._rag_agent is not None

        if has_rag:
            self._add_rag_nodes(builder)

        if self._movie_finder is not None:
            self._build_graph_with_movie_finder(builder, has_rag)
        else:
            self._build_graph_without_movie_finder(builder, has_rag)

        return builder.compile()

    def _create_orchestrate_node(self):
        """Create the appropriate orchestrate node based on available agents."""
        if self._input_agent is not None:
            return create_input_orchestrate_node(self._input_agent)
        elif self._orchestrator is not None:
            return create_orchestrate_node(self._orchestrator)
        else:
            raise ValueError("Either orchestrator or input_agent must be provided")

    def _add_rag_nodes(self, builder: StateGraph) -> None:
        """Add RAG retrieval and response nodes to the graph."""
        rag_retrieve_node = create_rag_retrieve_node(self._rag_retriever)
        rag_respond_node = create_rag_respond_node(self._rag_agent)
        builder.add_node("rag_retrieve", rag_retrieve_node)
        builder.add_node("rag_respond", rag_respond_node)

    def _build_graph_with_movie_finder(
        self, builder: StateGraph, has_rag: bool
    ) -> None:
        """Build graph edges when movie finder is available."""
        find_movies_node = create_find_movies_node(self._movie_finder)
        builder.add_node("find_movies", find_movies_node)

        builder.add_edge(START, "orchestrate")

        if has_rag:
            self._add_rag_enabled_edges(builder)
        else:
            builder.add_conditional_edges("orchestrate", route_after_orchestrate)

        if self._recommendation_writer is not None:
            self._add_recommendation_pipeline(builder, has_rag)
        else:
            builder.add_edge("find_movies", "respond")

        builder.add_edge("respond", END)

    def _add_rag_enabled_edges(self, builder: StateGraph) -> None:
        """Add edges for RAG-enabled workflow."""
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

    def _add_recommendation_pipeline(
        self, builder: StateGraph, has_rag: bool
    ) -> None:
        """Add recommendation writer and optional evaluator nodes."""
        write_node = create_write_recommendation_node(self._recommendation_writer)
        builder.add_node("write_recommendation", write_node)

        if has_rag:
            builder.add_conditional_edges(
                "find_movies",
                route_after_find_movies_for_hybrid,
                {
                    "rag_retrieve_hybrid": "rag_retrieve_hybrid",
                    "write_recommendation": "write_recommendation",
                },
            )
            rag_retrieve_hybrid_node = create_rag_retrieve_node(self._rag_retriever)
            builder.add_node("rag_retrieve_hybrid", rag_retrieve_hybrid_node)
            builder.add_edge("rag_retrieve_hybrid", "write_recommendation")
        else:
            builder.add_edge("find_movies", "write_recommendation")

        if self._evaluator is not None:
            self._add_evaluator_pipeline(builder)
        else:
            builder.add_edge("write_recommendation", "respond")

    def _add_evaluator_pipeline(self, builder: StateGraph) -> None:
        """Add evaluator node with retry loop."""
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

    def _build_graph_without_movie_finder(
        self, builder: StateGraph, has_rag: bool
    ) -> None:
        """Build graph edges when no movie finder is available."""
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
            "search_query": None,
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

    def get_response(
        self, user_message: str
    ) -> tuple[str, str | None, Constraints | None]:
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
