import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add the api/ directory to sys.path so "import app" works reliably
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.evaluator_agent import EvaluatorAgent, StubEvaluatorAgent
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import MovieFinderAgent, StubMovieFinderAgent
from app.llm.rag_agent import RAGAssistantAgent, StubRAGAssistantAgent
from app.llm.recommendation_agent import (
    RecommendationWriterAgent,
    StubRecommendationWriterAgent,
)
from app.rag.retriever import DocumentRetriever
from app.schemas.domain import MovieResult


@pytest.fixture
def mock_orchestrator():
    return MagicMock(spec=OrchestratorAgent)


@pytest.fixture
def mock_input_agent():
    return MagicMock(spec=InputOrchestratorAgent)


@pytest.fixture
def mock_movies_responder():
    return MagicMock(spec=MoviesResponder)


@pytest.fixture
def mock_system_responder():
    return MagicMock(spec=SystemResponder)


@pytest.fixture
def mock_movie_finder():
    return MagicMock(spec=MovieFinderAgent)


@pytest.fixture
def stub_movie_finder():
    return StubMovieFinderAgent()


@pytest.fixture
def mock_recommendation_writer():
    return MagicMock(spec=RecommendationWriterAgent)


@pytest.fixture
def stub_recommendation_writer():
    return StubRecommendationWriterAgent()


@pytest.fixture
def mock_evaluator():
    return MagicMock(spec=EvaluatorAgent)


@pytest.fixture
def stub_evaluator():
    return StubEvaluatorAgent()


@pytest.fixture
def mock_rag_retriever():
    return MagicMock(spec=DocumentRetriever)


@pytest.fixture
def mock_rag_agent():
    return MagicMock(spec=RAGAssistantAgent)


@pytest.fixture
def stub_rag_agent():
    return StubRAGAssistantAgent()


def make_movie(
    id_: str,
    title: str,
    genres: list[str] | None = None,
    rating: float | None = None,
    overview: str | None = None,
    runtime_minutes: int | None = None,
    year: int | None = None,
) -> MovieResult:
    """Factory function to create test MovieResult objects."""
    return MovieResult(
        id=id_,
        title=title,
        genres=genres or [],
        rating=rating,
        overview=overview,
        runtime_minutes=runtime_minutes,
        year=year,
        source="test",
    )