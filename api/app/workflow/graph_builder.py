"""LangGraph workflow builder for the Movie Night Assistant.

This module contains the MovieNightWorkflow class that constructs and
manages the LangGraph StateGraph for the recommendation workflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.workflow.state import MovieNightState
from app.workflow.nodes import (
    create_evaluate_node,
    create_find_movies_node,
    create_input_orchestrate_node,
    create_noop_rag_respond_node,
    create_noop_rag_retrieve_node,
    create_rag_respond_node,
    create_rag_retrieve_node,
    create_write_recommendation_node,
)
from app.workflow.routing import (
    route_after_evaluate,
    route_after_orchestrate,
    route_after_find_movies_for_hybrid,
)
from app.schemas.input import Constraints

if TYPE_CHECKING:
    from app.agents import EvaluatorAgent
    from app.agents import InputOrchestratorAgent
    from app.agents import RAGAssistantAgent
    from app.agents import RecommendationWriterAgent
    from app.agents import MovieFinderAgent
    from app.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class MovieNightWorkflow:
    """Wrapper class for the Movie Night Assistant LangGraph workflow.

    Encapsulates the graph construction and provides a simple interface
    for executing the workflow with a user message.

    RAG agents are optional — when not provided, RAG routes return a
    graceful fallback message and hybrid routes skip context retrieval.
    """

    def __init__(
        self,
        input_agent: InputOrchestratorAgent,
        movie_finder: MovieFinderAgent,
        rag_retriever: DocumentRetriever | None,
        rag_agent: RAGAssistantAgent | None,
        recommendation_writer: RecommendationWriterAgent,
        evaluator: EvaluatorAgent,
    ) -> None:
        """Initialize the workflow with agent instances.

        Args:
            input_agent: The InputOrchestratorAgent for route classification
                (movies/rag/hybrid).
            movie_finder: The MovieFinderAgent for candidate retrieval from TMDB or in-memory.
            rag_retriever: The DocumentRetriever for knowledge base retrieval (None to disable).
            rag_agent: The RAGAssistantAgent for grounded answers from docs (None to disable).
            recommendation_writer: The RecommendationWriterAgent for grounded prose.
            evaluator: The EvaluatorAgent for draft validation with retry loop.
        """
        if input_agent is None:
            raise ValueError("input_agent must be provided")
        self._input_agent = input_agent
        self._movie_finder = movie_finder
        self._rag_retriever = rag_retriever
        self._rag_agent = rag_agent
        self._recommendation_writer = recommendation_writer
        self._evaluator = evaluator
        self._rag_enabled = rag_retriever is not None and rag_agent is not None
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the workflow graph.

        The graph is fixed: RAG retrieval/response for system questions,
        movie candidate retrieval, recommendation writing, and evaluator
        retry loop are always present. Unknown routes default to clarification.

        Returns:
            Compiled StateGraph ready for execution.
        """
        builder = StateGraph(MovieNightState)

        builder.add_node("orchestrate", create_input_orchestrate_node(self._input_agent))
        if self._rag_enabled:
            builder.add_node("rag_retrieve", create_rag_retrieve_node(self._rag_retriever))
            builder.add_node("rag_respond", create_rag_respond_node(self._rag_agent))
        else:
            builder.add_node("rag_retrieve", create_noop_rag_retrieve_node())
            builder.add_node("rag_respond", create_noop_rag_respond_node())
        builder.add_node("find_movies", create_find_movies_node(self._movie_finder))

        builder.add_edge(START, "orchestrate")
        builder.add_conditional_edges(
            "orchestrate",
            route_after_orchestrate,
            {
                END: END,
                "find_movies": "find_movies",
                "rag_retrieve": "rag_retrieve",
            },
        )
        builder.add_edge("rag_retrieve", "rag_respond")
        builder.add_edge("rag_respond", END)

        self._add_recommendation_pipeline(builder)

        return builder.compile()

    def _add_recommendation_pipeline(self, builder: StateGraph) -> None:
        """Add recommendation writer, hybrid RAG retrieval, and evaluator."""
        builder.add_node(
            "write_recommendation",
            create_write_recommendation_node(self._recommendation_writer),
        )
        if self._rag_enabled:
            builder.add_node(
                "rag_retrieve_hybrid",
                create_rag_retrieve_node(self._rag_retriever),
            )
        else:
            builder.add_node(
                "rag_retrieve_hybrid",
                create_noop_rag_retrieve_node(),
            )

        builder.add_conditional_edges(
            "find_movies",
            route_after_find_movies_for_hybrid,
            {
                "rag_retrieve_hybrid": "rag_retrieve_hybrid",
                "write_recommendation": "write_recommendation",
            },
        )
        builder.add_edge("rag_retrieve_hybrid", "write_recommendation")
        self._add_evaluator_pipeline(builder)

    def _add_evaluator_pipeline(self, builder: StateGraph) -> None:
        """Add evaluator node with retry loop."""
        builder.add_node("evaluate", create_evaluate_node(self._evaluator))
        builder.add_edge("write_recommendation", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            route_after_evaluate,
            {
                END: END,
                "write_recommendation": "write_recommendation",
            },
        )

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
