"""AI agent implementations for the Movie Night Assistant.

All agents that interact with LLMs or external services live here:

- :class:`InputOrchestratorAgent`: classifies routes and extracts constraints
- :class:`MovieFinderAgent` / :class:`InMemoryMovieFinderAgent` / :class:`TMDBMovieFinderAgent`: retrieves candidate movies
- :class:`RecommendationWriterAgent` / :class:`LLMRecommendationWriterAgent`: composes recommendation drafts
- :class:`EvaluatorAgent` / :class:`LLMEvaluatorAgent`: validates draft recommendations
- :class:`RAGAssistantAgent` / :class:`LLMRAGAssistantAgent`: answers system questions using the knowledge base
- :class:`SystemResponder`: fallback responder for unexpected routes
"""

from app.agents.evaluator_agent import EvaluatorAgent, LLMEvaluatorAgent
from app.agents.input_agent import InputOrchestratorAgent
from app.agents.movie_finder_agent import MovieFinderAgent
from app.agents.in_memory_movie_finder_agent import InMemoryMovieFinderAgent
from app.agents.tmdb_movie_finder_agent import TMDBMovieFinderAgent
from app.agents.rag_agent import LLMRAGAssistantAgent, RAGAssistantAgent
from app.agents.recommendation_agent import (
    LLMRecommendationWriterAgent,
    RecommendationWriterAgent,
)
from app.agents.system_responder import SystemResponder

__all__ = [
    "EvaluatorAgent",
    "LLMEvaluatorAgent",
    "InputOrchestratorAgent",
    "MovieFinderAgent",
    "InMemoryMovieFinderAgent",
    "TMDBMovieFinderAgent",
    "LLMRAGAssistantAgent",
    "RAGAssistantAgent",
    "LLMRecommendationWriterAgent",
    "RecommendationWriterAgent",
    "SystemResponder",
]
