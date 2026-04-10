"""Unit tests for the LangGraph workflow."""

from unittest.mock import MagicMock

import pytest

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.workflow import (
    MovieNightWorkflow,
    create_orchestrate_node,
    create_respond_node,
    should_respond,
)
from app.schemas.orchestrator import Constraints, OrchestratorDecision


@pytest.fixture
def mock_orchestrator():
    return MagicMock(spec=OrchestratorAgent)


@pytest.fixture
def mock_movies_responder():
    return MagicMock(spec=MoviesResponder)


@pytest.fixture
def mock_system_responder():
    return MagicMock(spec=SystemResponder)


class TestOrchestrateNode:
    def test_orchestrate_routes_to_movies(self, mock_orchestrator):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(genres=["comedy"]),
            needs_clarification=False,
        )

        node = create_orchestrate_node(mock_orchestrator)
        result = node({"user_message": "Recommend a comedy"})

        assert result["route"] == "movies"
        assert result["constraints"].genres == ["comedy"]
        assert "final_response" not in result
        mock_orchestrator.decide.assert_called_once_with("Recommend a comedy")

    def test_orchestrate_routes_to_system(self, mock_orchestrator):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="system",
            constraints=Constraints(),
            needs_clarification=False,
        )

        node = create_orchestrate_node(mock_orchestrator)
        result = node({"user_message": "How does this work?"})

        assert result["route"] == "system"
        assert result["constraints"].genres == []

    def test_orchestrate_handles_clarification(self, mock_orchestrator):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What genre do you prefer?",
        )

        node = create_orchestrate_node(mock_orchestrator)
        result = node({"user_message": "help"})

        assert result["route"] == "clarification"
        assert result["final_response"] == "What genre do you prefer?"

    def test_orchestrate_default_clarification_message(self, mock_orchestrator):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question=None,
        )

        node = create_orchestrate_node(mock_orchestrator)
        result = node({"user_message": "?"})

        assert result["route"] == "clarification"
        assert "clarify" in result["final_response"].lower()


class TestRespondNode:
    def test_respond_movies_route(self, mock_movies_responder, mock_system_responder):
        mock_movies_responder.respond.return_value = "Here's a great comedy!"

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Recommend a comedy",
            "route": "movies",
            "constraints": Constraints(genres=["comedy"]),
        })

        assert result["final_response"] == "Here's a great comedy!"
        mock_movies_responder.respond.assert_called_once()
        mock_system_responder.respond.assert_not_called()

    def test_respond_system_route(self, mock_movies_responder, mock_system_responder):
        mock_system_responder.respond.return_value = "This app helps you find movies."

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "What does this app do?",
            "route": "system",
            "constraints": None,
        })

        assert result["final_response"] == "This app helps you find movies."
        mock_system_responder.respond.assert_called_once()
        mock_movies_responder.respond.assert_not_called()

    def test_respond_skips_clarification_route(
        self, mock_movies_responder, mock_system_responder
    ):
        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "help",
            "route": "clarification",
            "constraints": None,
        })

        assert result == {}
        mock_movies_responder.respond.assert_not_called()
        mock_system_responder.respond.assert_not_called()


class TestShouldRespond:
    def test_should_respond_movies(self):
        assert should_respond({"route": "movies"}) == "respond"

    def test_should_respond_system(self):
        assert should_respond({"route": "system"}) == "respond"

    def test_should_not_respond_clarification(self):
        from langgraph.graph import END
        assert should_respond({"route": "clarification"}) == END


class TestMovieNightWorkflow:
    def test_workflow_movies_happy_path(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
        )
        mock_movies_responder.respond.return_value = "Check out these action movies!"

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert result["final_response"] == "Check out these action movies!"
        assert result["constraints"].genres == ["action"]

    def test_workflow_system_happy_path(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="system",
            constraints=Constraints(),
            needs_clarification=False,
        )
        mock_system_responder.respond.return_value = "I help you find movies."

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder
        )
        result = workflow.invoke("How do you work?")

        assert result["route"] == "system"
        assert result["final_response"] == "I help you find movies."

    def test_workflow_clarification_path(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What mood are you in?",
        )

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder
        )
        result = workflow.invoke("something")

        assert result["route"] == "clarification"
        assert result["final_response"] == "What mood are you in?"
        mock_movies_responder.respond.assert_not_called()
        mock_system_responder.respond.assert_not_called()

    def test_get_response_extracts_fields(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(genres=["horror"], max_runtime_minutes=120),
            needs_clarification=False,
        )
        mock_movies_responder.respond.return_value = "Try The Conjuring!"

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder
        )
        reply, route, constraints = workflow.get_response("Short horror movie")

        assert reply == "Try The Conjuring!"
        assert route == "movies"
        assert constraints.genres == ["horror"]
        assert constraints.max_runtime_minutes == 120

    def test_get_response_raises_on_no_response(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(),
            needs_clarification=False,
        )
        mock_movies_responder.respond.return_value = None

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder
        )

        with pytest.raises(RuntimeError, match="did not produce a response"):
            workflow.get_response("Test")
