"""Fallback agent implementations.

These agents provide basic orchestration and response generation used
as fallbacks in the workflow when more specialized agents are not available
or when graceful degradation is needed.

- :class:`OrchestratorAgent`: Basic intent classifier (movies/system routes only).
  Use :class:`app.llm.input_agent.InputOrchestratorAgent` for full routing.

- :class:`MoviesResponder`: Fallback responder for movie routes when no draft
  recommendation is available.

- :class:`SystemResponder`: Fallback responder for system questions when the
  RAG pipeline is not configured.
"""

from app.agents.orchestrator import OrchestratorAgent
from app.agents.responder import MoviesResponder, SystemResponder

__all__ = [
    "OrchestratorAgent",
    "MoviesResponder",
    "SystemResponder",
]

