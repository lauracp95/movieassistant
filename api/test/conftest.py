import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_openai import AzureChatOpenAI

# Add the api/ directory to sys.path so "import app" works reliably
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import (
    EvaluatorAgent,
    InputOrchestratorAgent,
    InMemoryMovieFinderAgent,
    MovieFinderAgent,
    RAGAssistantAgent,
    RecommendationWriterAgent,
)
from app.rag.retriever import DocumentRetriever
from app.schemas.domain import EvaluationResult, MovieResult


@pytest.fixture
def mock_input_agent():
    return MagicMock(spec=InputOrchestratorAgent)


@pytest.fixture
def mock_movie_finder():
    return MagicMock(spec=MovieFinderAgent)


@pytest.fixture
def in_memory_movie_finder():
    return InMemoryMovieFinderAgent()


@pytest.fixture
def mock_recommendation_writer():
    return MagicMock(spec=RecommendationWriterAgent)


@pytest.fixture
def recommendation_writer():
    llm = MagicMock(spec=AzureChatOpenAI)
    llm.invoke.return_value = MagicMock(
        content="A great pick for your movie night."
    )
    return RecommendationWriterAgent(llm)


@pytest.fixture
def mock_evaluator():
    mock = MagicMock(spec=EvaluatorAgent)
    mock.evaluate.return_value = EvaluationResult(
        passed=True,
        score=1.0,
        feedback="Looks good.",
        constraint_violations=[],
        improvement_suggestions=[],
    )
    return mock


@pytest.fixture
def evaluator():
    llm = MagicMock(spec=AzureChatOpenAI)
    structured = MagicMock()
    structured.invoke.return_value = EvaluationResult(
        passed=True,
        score=0.85,
        feedback="Draft satisfies hard constraints.",
        constraint_violations=[],
        improvement_suggestions=[],
    )
    llm.with_structured_output.return_value = structured
    return EvaluatorAgent(llm)


@pytest.fixture
def mock_rag_retriever():
    return MagicMock(spec=DocumentRetriever)


@pytest.fixture
def mock_rag_agent():
    return MagicMock(spec=RAGAssistantAgent)


@pytest.fixture
def rag_agent():
    llm = MagicMock(spec=AzureChatOpenAI)
    llm.invoke.return_value = MagicMock(
        content=(
            "Based on my knowledge base, I can help answer "
            "questions about how the Movie Night Assistant works."
        )
    )
    return RAGAssistantAgent(llm)


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
