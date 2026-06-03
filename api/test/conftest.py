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
    LLMEvaluatorAgent,
    LLMRAGAssistantAgent,
    MovieFinderAgent,
    RAGAssistantAgent,
    InMemoryMovieFinderAgent,
    SystemResponder,
)
from app.agents.recommendation_agent import (
    LLMRecommendationWriterAgent,
    RecommendationWriterAgent,
)
from app.rag.retriever import DocumentRetriever
from app.schemas.domain import EvaluationResult, MovieResult


@pytest.fixture
def mock_input_agent():
    return MagicMock(spec=InputOrchestratorAgent)


@pytest.fixture
def mock_system_responder():
    return MagicMock(spec=SystemResponder)


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
def recommendation_writer_llm():
    llm = MagicMock(spec=AzureChatOpenAI)
    llm.invoke.return_value = MagicMock(
        content="A great pick for your movie night."
    )
    return llm


@pytest.fixture
def llm_recommendation_writer(recommendation_writer_llm):
    return LLMRecommendationWriterAgent(recommendation_writer_llm)


@pytest.fixture
def mock_evaluator():
    return MagicMock(spec=EvaluatorAgent)


@pytest.fixture
def evaluator_llm():
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
    return llm


@pytest.fixture
def llm_evaluator(evaluator_llm):
    return LLMEvaluatorAgent(evaluator_llm)


@pytest.fixture
def mock_rag_retriever():
    return MagicMock(spec=DocumentRetriever)


@pytest.fixture
def mock_rag_agent():
    return MagicMock(spec=RAGAssistantAgent)


@pytest.fixture
def rag_agent_llm():
    llm = MagicMock(spec=AzureChatOpenAI)
    llm.invoke.return_value = MagicMock(
        content=(
            "Based on my knowledge base, I can help answer "
            "questions about how the Movie Night Assistant works."
        )
    )
    return llm


@pytest.fixture
def llm_rag_agent(rag_agent_llm):
    return LLMRAGAssistantAgent(rag_agent_llm)


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
