"""LangGraph workflow package for the Movie Night Assistant.

This package contains the workflow implementation split into focused modules:

- :mod:`nodes`: Node creation functions for each workflow step
- :mod:`routing`: Routing decision functions for conditional edges
- :mod:`formatters`: Response formatting utilities
- :mod:`graph_builder`: The main MovieNightWorkflow class

Example usage::

    from app.llm.workflow import MovieNightWorkflow
    
    workflow = MovieNightWorkflow(
        orchestrator=None,
        movies_responder=movies_responder,
        system_responder=system_responder,
        input_agent=input_agent,
        movie_finder=movie_finder,
    )
    
    result = workflow.invoke("Recommend a comedy movie")
"""

from app.llm.workflow.formatters import (
    NO_MOVIES_FOUND_MESSAGE,
    RETRY_EXHAUSTED_FALLBACK_MESSAGE,
    format_candidate_list_response,
)
from app.llm.workflow.graph_builder import MovieNightWorkflow
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

__all__ = [
    "MovieNightWorkflow",
    "create_orchestrate_node",
    "create_input_orchestrate_node",
    "create_respond_node",
    "create_find_movies_node",
    "create_write_recommendation_node",
    "create_evaluate_node",
    "create_rag_retrieve_node",
    "create_rag_respond_node",
    "route_after_evaluate",
    "route_after_orchestrate",
    "route_after_orchestrate_with_rag",
    "route_after_find_movies_for_hybrid",
    "should_respond",
    "format_candidate_list_response",
    "NO_MOVIES_FOUND_MESSAGE",
    "RETRY_EXHAUSTED_FALLBACK_MESSAGE",
]
