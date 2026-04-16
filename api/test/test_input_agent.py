"""Unit tests for the InputOrchestratorAgent."""

from unittest.mock import MagicMock, patch

import pytest

from app.llm.input_agent import InputOrchestratorAgent
from app.schemas.orchestrator import Constraints, InputDecision


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def input_agent(mock_llm):
    with patch.object(mock_llm, "with_structured_output") as mock_with_output:
        mock_with_output.return_value = mock_llm
        agent = InputOrchestratorAgent(mock_llm)
        agent._llm = mock_llm
        return agent


class TestInputOrchestratorAgentClassification:
    def test_classifies_movie_request_correctly(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["comedy"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        decision = input_agent.decide("Recommend a comedy movie")

        assert decision.route == "movies"
        assert decision.needs_recommendation is True
        assert decision.rag_query is None
        assert "comedy" in decision.constraints.genres

    def test_classifies_rag_request_correctly(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does the Movie Night Assistant work?",
        )

        decision = input_agent.decide("How does this app work?")

        assert decision.route == "rag"
        assert decision.needs_recommendation is False
        assert decision.rag_query is not None
        assert "Movie Night" in decision.rag_query or "work" in decision.rag_query

    def test_classifies_hybrid_request_correctly(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History and traditions of Halloween horror movies",
        )

        decision = input_agent.decide(
            "What horror movies are best for Halloween and what's the history?"
        )

        assert decision.route == "hybrid"
        assert decision.needs_recommendation is True
        assert decision.rag_query is not None
        assert "horror" in decision.constraints.genres


class TestInputOrchestratorAgentClarification:
    def test_detects_clarification_needed(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="Are you looking for movie recommendations or do you have a question about the app?",
            needs_recommendation=False,
            rag_query=None,
        )

        decision = input_agent.decide("help")

        assert decision.needs_clarification is True
        assert decision.clarification_question is not None

    def test_handles_vague_message(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What kind of movie are you in the mood for?",
            needs_recommendation=False,
            rag_query=None,
        )

        decision = input_agent.decide("hi")

        assert decision.needs_clarification is True
        assert decision.clarification_question is not None


class TestInputOrchestratorAgentConstraints:
    def test_extracts_genre_constraints(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["horror", "thriller"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        decision = input_agent.decide("I want a scary thriller movie")

        assert "horror" in decision.constraints.genres or "thriller" in decision.constraints.genres

    def test_extracts_runtime_constraints(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["comedy"], max_runtime_minutes=120),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        decision = input_agent.decide("Recommend a comedy under 2 hours")

        assert decision.constraints.max_runtime_minutes == 120
        assert "comedy" in decision.constraints.genres

    def test_extracts_min_runtime_constraints(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["drama"], min_runtime_minutes=120),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        decision = input_agent.decide("I want a long drama movie, over 2 hours")

        assert decision.constraints.min_runtime_minutes == 120
        assert "drama" in decision.constraints.genres


class TestInputOrchestratorAgentValidation:
    def test_validates_movies_route_sets_needs_recommendation(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="should be cleared",
        )

        decision = input_agent.decide("Recommend an action movie")

        assert decision.route == "movies"
        assert decision.needs_recommendation is True
        assert decision.rag_query is None

    def test_validates_rag_route_clears_needs_recommendation(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="How does this work?",
        )

        decision = input_agent.decide("How does this work?")

        assert decision.route == "rag"
        assert decision.needs_recommendation is False

    def test_validates_hybrid_route_keeps_both(self, input_agent, mock_llm):
        mock_llm.invoke.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["romance"]),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="What makes good date night movies",
        )

        decision = input_agent.decide("What's a good date night movie and why?")

        assert decision.route == "hybrid"
        assert decision.needs_recommendation is True


class TestInputDecisionSchema:
    def test_input_decision_default_values(self):
        decision = InputDecision(route="movies")

        assert decision.route == "movies"
        assert decision.constraints.genres == []
        assert decision.constraints.max_runtime_minutes is None
        assert decision.constraints.min_runtime_minutes is None
        assert decision.confidence is None
        assert decision.needs_clarification is False
        assert decision.clarification_question is None
        assert decision.needs_recommendation is True
        assert decision.rag_query is None

    def test_input_decision_with_all_fields(self):
        decision = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["sci-fi"], max_runtime_minutes=150),
            confidence=0.85,
            needs_clarification=False,
            clarification_question=None,
            needs_recommendation=True,
            rag_query="Science fiction movie themes and history",
        )

        assert decision.route == "hybrid"
        assert decision.constraints.genres == ["sci-fi"]
        assert decision.constraints.max_runtime_minutes == 150
        assert decision.confidence == 0.85
        assert decision.needs_recommendation is True
        assert decision.rag_query == "Science fiction movie themes and history"

    def test_input_decision_validates_route(self):
        with pytest.raises(ValueError):
            InputDecision(route="invalid_route")

    def test_input_decision_validates_confidence_range(self):
        with pytest.raises(ValueError):
            InputDecision(route="movies", confidence=1.5)

        with pytest.raises(ValueError):
            InputDecision(route="movies", confidence=-0.1)
