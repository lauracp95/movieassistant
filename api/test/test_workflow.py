"""Unit tests for the LangGraph workflow."""

from unittest.mock import MagicMock

import pytest

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import MovieFinderAgent, StubMovieFinderAgent
from app.llm.workflow import (
    MovieNightWorkflow,
    create_find_movies_node,
    create_input_orchestrate_node,
    create_orchestrate_node,
    create_respond_node,
    route_after_orchestrate,
    should_respond,
)
from app.schemas.domain import MovieResult
from app.schemas.orchestrator import Constraints, InputDecision, OrchestratorDecision


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
    def test_respond_movies_route_with_candidates(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [
            MovieResult(
                id="test-1",
                title="Action Movie",
                year=2023,
                genres=["Action"],
                overview="A great action movie.",
                rating=8.0,
                source="test",
            ),
        ]

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Recommend action movies",
            "route": "movies",
            "constraints": Constraints(genres=["action"]),
            "candidate_movies": candidates,
        })

        assert "final_response" in result
        assert "Action Movie" in result["final_response"]
        mock_movies_responder.respond.assert_not_called()

    def test_respond_movies_route_no_candidates(
        self, mock_movies_responder, mock_system_responder
    ):
        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Recommend a rare genre movie",
            "route": "movies",
            "constraints": Constraints(),
            "candidate_movies": [],
        })

        assert "final_response" in result
        assert "couldn't find" in result["final_response"].lower()

    def test_respond_rag_route(self, mock_movies_responder, mock_system_responder):
        mock_system_responder.respond.return_value = "This app helps you find movies."

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "What does this app do?",
            "route": "rag",
            "constraints": None,
            "candidate_movies": [],
        })

        assert result["final_response"] == "This app helps you find movies."
        mock_system_responder.respond.assert_called_once()

    def test_respond_hybrid_route_with_candidates(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [
            MovieResult(
                id="test-1",
                title="Horror Classic",
                year=1990,
                genres=["Horror"],
                overview="A classic horror film.",
                rating=7.5,
                source="test",
            ),
        ]

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Horror movies and their history",
            "route": "hybrid",
            "constraints": Constraints(genres=["horror"]),
            "candidate_movies": candidates,
        })

        assert "Horror Classic" in result["final_response"]

    def test_respond_skips_clarification_route(
        self, mock_movies_responder, mock_system_responder
    ):
        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "help",
            "route": "clarification",
            "constraints": None,
            "candidate_movies": [],
        })

        assert result == {}
        mock_movies_responder.respond.assert_not_called()
        mock_system_responder.respond.assert_not_called()

    def test_respond_system_route_fallback(
        self, mock_movies_responder, mock_system_responder
    ):
        mock_system_responder.respond.return_value = "Fallback response"

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Unknown route test",
            "route": "unknown",
            "constraints": None,
            "candidate_movies": [],
        })

        assert result["final_response"] == "Fallback response"
        mock_system_responder.respond.assert_called_once()


class TestInputOrchestrateNode:
    def test_input_orchestrate_routes_to_movies(self, mock_input_agent):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["comedy"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        node = create_input_orchestrate_node(mock_input_agent)
        result = node({"user_message": "Recommend a comedy"})

        assert result["route"] == "movies"
        assert result["constraints"].genres == ["comedy"]
        assert result["needs_recommendation"] is True
        assert result["rag_query"] is None
        assert "final_response" not in result
        mock_input_agent.decide.assert_called_once_with("Recommend a comedy")

    def test_input_orchestrate_routes_to_rag(self, mock_input_agent):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does this app work?",
        )

        node = create_input_orchestrate_node(mock_input_agent)
        result = node({"user_message": "How does this work?"})

        assert result["route"] == "rag"
        assert result["needs_recommendation"] is False
        assert result["rag_query"] == "How does this app work?"

    def test_input_orchestrate_routes_to_hybrid(self, mock_input_agent):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of Halloween horror movies",
        )

        node = create_input_orchestrate_node(mock_input_agent)
        result = node({"user_message": "Horror movies for Halloween and their history"})

        assert result["route"] == "hybrid"
        assert result["constraints"].genres == ["horror"]
        assert result["needs_recommendation"] is True
        assert result["rag_query"] == "History of Halloween horror movies"

    def test_input_orchestrate_handles_clarification(self, mock_input_agent):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What genre do you prefer?",
            needs_recommendation=False,
            rag_query=None,
        )

        node = create_input_orchestrate_node(mock_input_agent)
        result = node({"user_message": "help"})

        assert result["route"] == "clarification"
        assert result["final_response"] == "What genre do you prefer?"
        assert result["needs_recommendation"] is False

    def test_input_orchestrate_default_clarification_message(self, mock_input_agent):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question=None,
            needs_recommendation=False,
            rag_query=None,
        )

        node = create_input_orchestrate_node(mock_input_agent)
        result = node({"user_message": "?"})

        assert result["route"] == "clarification"
        assert "clarify" in result["final_response"].lower()


class TestShouldRespond:
    def test_should_respond_movies(self):
        assert should_respond({"route": "movies"}) == "respond"

    def test_should_respond_system(self):
        assert should_respond({"route": "system"}) == "respond"

    def test_should_respond_rag(self):
        assert should_respond({"route": "rag"}) == "respond"

    def test_should_respond_hybrid(self):
        assert should_respond({"route": "hybrid"}) == "respond"

    def test_should_not_respond_clarification(self):
        from langgraph.graph import END
        assert should_respond({"route": "clarification"}) == END


class TestRouteAfterOrchestrate:
    def test_routes_to_find_movies_for_movies_route(self):
        assert route_after_orchestrate({"route": "movies"}) == "find_movies"

    def test_routes_to_find_movies_for_hybrid_route(self):
        assert route_after_orchestrate({"route": "hybrid"}) == "find_movies"

    def test_routes_to_respond_for_rag_route(self):
        assert route_after_orchestrate({"route": "rag"}) == "respond"

    def test_routes_to_end_for_clarification(self):
        from langgraph.graph import END
        assert route_after_orchestrate({"route": "clarification"}) == END


class TestFindMoviesNode:
    def test_find_movies_node_populates_candidates(self, mock_movie_finder):
        mock_movie_finder.find_movies.return_value = [
            MovieResult(
                id="test-1",
                title="Test Movie",
                genres=["Action"],
                source="test",
            ),
        ]

        node = create_find_movies_node(mock_movie_finder)
        result = node({
            "constraints": Constraints(genres=["Action"]),
            "rejected_titles": [],
        })

        assert "candidate_movies" in result
        assert len(result["candidate_movies"]) == 1
        assert result["candidate_movies"][0].title == "Test Movie"
        mock_movie_finder.find_movies.assert_called_once()

    def test_find_movies_node_passes_constraints(self, mock_movie_finder):
        mock_movie_finder.find_movies.return_value = []

        node = create_find_movies_node(mock_movie_finder)
        constraints = Constraints(genres=["Horror"], max_runtime_minutes=120)
        node({
            "constraints": constraints,
            "rejected_titles": ["Bad Movie"],
        })

        mock_movie_finder.find_movies.assert_called_once_with(
            constraints=constraints,
            limit=10,
            excluded_titles=["Bad Movie"],
        )

    def test_find_movies_node_handles_empty_results(self, mock_movie_finder):
        mock_movie_finder.find_movies.return_value = []

        node = create_find_movies_node(mock_movie_finder)
        result = node({
            "constraints": Constraints(),
            "rejected_titles": [],
        })

        assert result["candidate_movies"] == []

    def test_find_movies_node_default_constraints(self, mock_movie_finder):
        mock_movie_finder.find_movies.return_value = []

        node = create_find_movies_node(mock_movie_finder)
        node({"constraints": None})

        call_args = mock_movie_finder.find_movies.call_args
        assert call_args[1]["constraints"] == Constraints()


class TestMovieNightWorkflow:
    def test_workflow_movies_happy_path_with_finder(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder, stub_movie_finder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
        )

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder,
            movie_finder=stub_movie_finder,
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert len(result["candidate_movies"]) > 0
        assert "final_response" in result

    def test_workflow_movies_without_finder_shows_no_results(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
        )

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert result["candidate_movies"] == []
        assert "couldn't find" in result["final_response"].lower()

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
        self, mock_orchestrator, mock_movies_responder, mock_system_responder, stub_movie_finder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(genres=["horror"], max_runtime_minutes=120),
            needs_clarification=False,
        )

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder,
            movie_finder=stub_movie_finder,
        )
        reply, route, constraints = workflow.get_response("Short horror movie")

        assert route == "movies"
        assert constraints.genres == ["horror"]
        assert constraints.max_runtime_minutes == 120
        assert reply is not None

    def test_get_response_raises_on_no_response(
        self, mock_orchestrator, mock_movies_responder, mock_system_responder, mock_movie_finder
    ):
        mock_orchestrator.decide.return_value = OrchestratorDecision(
            intent="movies",
            constraints=Constraints(),
            needs_clarification=False,
        )
        mock_movie_finder.find_movies.return_value = []

        workflow = MovieNightWorkflow(
            mock_orchestrator, mock_movies_responder, mock_system_responder,
            movie_finder=mock_movie_finder,
        )
        reply, route, constraints = workflow.get_response("Test")

        assert "couldn't find" in reply.lower()


class TestMovieNightWorkflowWithInputAgent:
    def test_workflow_movies_with_input_agent(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, stub_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=stub_movie_finder,
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert result["constraints"].genres == ["action"]
        assert result["needs_recommendation"] is True
        assert len(result["candidate_movies"]) > 0

    def test_workflow_rag_with_input_agent(
        self, mock_input_agent, mock_movies_responder, mock_system_responder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does the app work?",
        )
        mock_system_responder.respond.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
        )
        result = workflow.invoke("How do you work?")

        assert result["route"] == "rag"
        assert result["final_response"] == "This app helps you find movies."
        assert result["needs_recommendation"] is False
        assert result["rag_query"] == "How does the app work?"

    def test_workflow_hybrid_with_input_agent(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, stub_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of Halloween horror films",
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=stub_movie_finder,
        )
        result = workflow.invoke("Horror movies for Halloween and their history")

        assert result["route"] == "hybrid"
        assert result["constraints"].genres == ["horror"]
        assert result["needs_recommendation"] is True
        assert result["rag_query"] == "History of Halloween horror films"
        assert len(result["candidate_movies"]) > 0

    def test_workflow_clarification_with_input_agent(
        self, mock_input_agent, mock_movies_responder, mock_system_responder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What mood are you in?",
            needs_recommendation=False,
            rag_query=None,
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
        )
        result = workflow.invoke("something")

        assert result["route"] == "clarification"
        assert result["final_response"] == "What mood are you in?"
        mock_movies_responder.respond.assert_not_called()
        mock_system_responder.respond.assert_not_called()

    def test_workflow_requires_orchestrator_or_input_agent(
        self, mock_movies_responder, mock_system_responder
    ):
        with pytest.raises(ValueError, match="Either orchestrator or input_agent"):
            MovieNightWorkflow(
                orchestrator=None,
                movies_responder=mock_movies_responder,
                system_responder=mock_system_responder,
                input_agent=None,
            )

    def test_get_response_with_input_agent(
        self, mock_input_agent, mock_movies_responder, mock_system_responder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does the app work?",
        )
        mock_system_responder.respond.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
        )
        reply, route, constraints = workflow.get_response("How does this work?")

        assert reply == "This app helps you find movies."
        assert route == "rag"


class TestMovieNightWorkflowWithMovieFinder:
    def test_workflow_movies_with_finder(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, mock_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            MovieResult(
                id="test-1",
                title="Action Hero",
                year=2023,
                genres=["Action"],
                overview="An action-packed adventure.",
                rating=8.5,
                source="test",
            ),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert len(result["candidate_movies"]) == 1
        assert result["candidate_movies"][0].title == "Action Hero"
        assert "Action Hero" in result["final_response"]
        mock_movie_finder.find_movies.assert_called_once()

    def test_workflow_hybrid_with_finder(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, mock_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of horror films",
        )
        mock_movie_finder.find_movies.return_value = [
            MovieResult(
                id="test-2",
                title="Scary Movie",
                year=2020,
                genres=["Horror"],
                overview="A terrifying experience.",
                rating=7.0,
                source="test",
            ),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
        )
        result = workflow.invoke("Horror movies and their history")

        assert result["route"] == "hybrid"
        assert len(result["candidate_movies"]) == 1
        assert "Scary Movie" in result["final_response"]
        mock_movie_finder.find_movies.assert_called_once()

    def test_workflow_rag_skips_finder(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, mock_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does this work?",
        )
        mock_system_responder.respond.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
        )
        result = workflow.invoke("How does this work?")

        assert result["route"] == "rag"
        assert result["final_response"] == "This app helps you find movies."
        mock_movie_finder.find_movies.assert_not_called()

    def test_workflow_with_stub_finder_integration(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, stub_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=stub_movie_finder,
        )
        result = workflow.invoke("Horror movies please")

        assert result["route"] == "movies"
        assert len(result["candidate_movies"]) > 0
        for movie in result["candidate_movies"]:
            assert any("horror" in g.lower() for g in movie.genres)

    def test_workflow_empty_finder_results(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, mock_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["nonexistent"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = []

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
        )
        result = workflow.invoke("Recommend nonexistent genre movies")

        assert result["route"] == "movies"
        assert result["candidate_movies"] == []
        assert "couldn't find" in result["final_response"].lower()

    def test_workflow_clarification_skips_finder(
        self, mock_input_agent, mock_movies_responder, mock_system_responder, mock_movie_finder
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What genre do you prefer?",
            needs_recommendation=False,
            rag_query=None,
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
        )
        result = workflow.invoke("recommend something")

        assert result["route"] == "clarification"
        assert result["final_response"] == "What genre do you prefer?"
        mock_movie_finder.find_movies.assert_not_called()
