"""AI agent implementations for the Movie Night Assistant.

All agents that interact with LLMs or external services live here:

- :class:`InputOrchestratorAgent`: classifies routes and extracts constraints
- :class:`MovieFinderAgent` / :class:`StubMovieFinderAgent` / :class:`TMDBMovieFinderAgent`: retrieves candidate movies
- :class:`RecommendationWriterAgent` / :class:`StubRecommendationWriterAgent` / :class:`LLMRecommendationWriterAgent`: composes recommendation drafts
- :class:`EvaluatorAgent` / :class:`StubEvaluatorAgent` / :class:`LLMEvaluatorAgent`: validates draft recommendations
- :class:`RAGAssistantAgent` / :class:`StubRAGAssistantAgent` / :class:`LLMRAGAssistantAgent`: answers system questions using the knowledge base
- :class:`SystemResponder`: fallback responder for unexpected routes
"""

from app.agents.evaluator_agent import (
    EvaluatorAgent,
    LLMEvaluatorAgent,
    StubEvaluatorAgent,
)
from app.agents.input_agent import InputOrchestratorAgent
from app.agents.movie_finder_agent import (
    MovieFinderAgent,
    StubMovieFinderAgent,
    TMDBMovieFinderAgent,
)
from app.agents.rag_agent import (
    LLMRAGAssistantAgent,
    RAGAssistantAgent,
    StubRAGAssistantAgent,
)
from app.agents.recommendation_agent import (
    LLMRecommendationWriterAgent,
    RecommendationWriterAgent,
    StubRecommendationWriterAgent,
)
from app.agents.system_responder import SystemResponder

__all__ = [
    "EvaluatorAgent",
    "LLMEvaluatorAgent",
    "StubEvaluatorAgent",
    "InputOrchestratorAgent",
    "MovieFinderAgent",
    "StubMovieFinderAgent",
    "TMDBMovieFinderAgent",
    "LLMRAGAssistantAgent",
    "RAGAssistantAgent",
    "StubRAGAssistantAgent",
    "LLMRecommendationWriterAgent",
    "RecommendationWriterAgent",
    "StubRecommendationWriterAgent",
    "SystemResponder",
]
