"""Unit tests for the LangGraph workflow."""

from unittest.mock import MagicMock

import pytest

from app.agents import MoviesResponder, OrchestratorAgent, SystemResponder
from app.llm.evaluator_agent import EvaluatorAgent, StubEvaluatorAgent
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import MovieFinderAgent, StubMovieFinderAgent
from app.llm.recommendation_agent import (
    RecommendationWriterAgent,
    StubRecommendationWriterAgent,
)
from app.llm.state import MAX_RETRIES
from app.llm.workflow import (
    RETRY_EXHAUSTED_FALLBACK_MESSAGE,
    MovieNightWorkflow,
    create_evaluate_node,
    create_find_movies_node,
    create_input_orchestrate_node,
    create_orchestrate_node,
    create_respond_node,
    create_write_recommendation_node,
    route_after_evaluate,
    route_after_orchestrate,
    should_respond,
)
from app.schemas.domain import (
    DraftRecommendation,
    EvaluationResult,
    MovieResult,
)
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


def _candidate(
    id_: str,
    title: str,
    genres: list[str] | None = None,
    rating: float | None = None,
    overview: str | None = None,
    runtime_minutes: int | None = None,
) -> MovieResult:
    return MovieResult(
        id=id_,
        title=title,
        genres=genres or [],
        rating=rating,
        overview=overview,
        runtime_minutes=runtime_minutes,
        source="test",
    )


class TestWriteRecommendationNode:
    def test_skips_when_no_candidates(self, mock_recommendation_writer):
        node = create_write_recommendation_node(mock_recommendation_writer)

        result = node({
            "user_message": "Recommend something",
            "constraints": Constraints(),
            "candidate_movies": [],
            "rejected_titles": [],
        })

        assert result == {"draft_recommendation": None}
        mock_recommendation_writer.write.assert_not_called()

    def test_populates_draft_recommendation(self, mock_recommendation_writer):
        movie = _candidate("1", "The Matrix", genres=["Sci-Fi"], rating=8.7)
        draft = DraftRecommendation(
            movie=movie,
            recommendation_text="Watch The Matrix, it's a classic.",
            reasoning="matches genres: sci-fi; rating 8.7/10",
        )
        mock_recommendation_writer.write.return_value = draft

        node = create_write_recommendation_node(mock_recommendation_writer)
        result = node({
            "user_message": "Sci-fi please",
            "constraints": Constraints(genres=["sci-fi"]),
            "candidate_movies": [movie],
            "rejected_titles": [],
        })

        assert result["draft_recommendation"] is draft
        mock_recommendation_writer.write.assert_called_once()
        kwargs = mock_recommendation_writer.write.call_args.kwargs
        assert kwargs["user_message"] == "Sci-fi please"
        assert kwargs["constraints"].genres == ["sci-fi"]
        assert kwargs["candidates"] == [movie]
        assert kwargs["rejected_titles"] == []

    def test_forwards_rejected_titles(self, mock_recommendation_writer):
        mock_recommendation_writer.write.return_value = None

        node = create_write_recommendation_node(mock_recommendation_writer)
        node({
            "user_message": "Sci-fi please",
            "constraints": Constraints(),
            "candidate_movies": [_candidate("1", "Any")],
            "rejected_titles": ["Bad Movie"],
        })

        kwargs = mock_recommendation_writer.write.call_args.kwargs
        assert kwargs["rejected_titles"] == ["Bad Movie"]

    def test_handles_writer_returning_none(self, mock_recommendation_writer):
        mock_recommendation_writer.write.return_value = None

        node = create_write_recommendation_node(mock_recommendation_writer)
        result = node({
            "user_message": "Anything",
            "constraints": Constraints(),
            "candidate_movies": [_candidate("1", "Only")],
            "rejected_titles": ["Only"],
        })

        assert result == {"draft_recommendation": None}


class TestRespondNodeWithDraft:
    def test_respond_uses_draft_text_when_available(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [_candidate("1", "Fallback Title", genres=["Action"])]
        draft = DraftRecommendation(
            movie=_candidate(
                "2",
                "Selected Movie",
                genres=["Action"],
                rating=8.0,
                overview="A selected story.",
            ),
            recommendation_text="Selected Movie is a grounded pick for action fans.",
            reasoning="matches genres: action",
        )

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Action movie",
            "route": "movies",
            "constraints": Constraints(genres=["action"]),
            "candidate_movies": candidates,
            "draft_recommendation": draft,
        })

        assert result["final_response"] == (
            "Selected Movie is a grounded pick for action fans."
        )
        assert "Fallback Title" not in result["final_response"]

    def test_respond_falls_back_to_candidates_without_draft(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [_candidate("1", "Fallback Title", genres=["Action"])]

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Action movie",
            "route": "movies",
            "constraints": Constraints(genres=["action"]),
            "candidate_movies": candidates,
            "draft_recommendation": None,
        })

        assert "Fallback Title" in result["final_response"]

    def test_respond_fallback_excludes_rejected_titles(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [
            _candidate("1", "Rejected Pick", genres=["Action"]),
            _candidate("2", "Safe Pick", genres=["Action"]),
        ]

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Action movie",
            "route": "movies",
            "constraints": Constraints(genres=["action"]),
            "candidate_movies": candidates,
            "rejected_titles": ["Rejected Pick"],
            "draft_recommendation": None,
        })

        assert "Rejected Pick" not in result["final_response"]
        assert "Safe Pick" in result["final_response"]

    def test_respond_fallback_excludes_runtime_violations(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [
            _candidate("1", "Too Long", genres=["Action"], runtime_minutes=200),
            _candidate("2", "Just Right", genres=["Action"], runtime_minutes=90),
        ]

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Short action movie",
            "route": "movies",
            "constraints": Constraints(
                genres=["action"], max_runtime_minutes=120
            ),
            "candidate_movies": candidates,
            "rejected_titles": [],
            "draft_recommendation": None,
        })

        assert "Too Long" not in result["final_response"]
        assert "Just Right" in result["final_response"]

    def test_respond_fallback_shows_no_match_when_all_rejected(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [_candidate("1", "Only", genres=["Action"])]

        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Action movie",
            "route": "movies",
            "constraints": Constraints(genres=["action"]),
            "candidate_movies": candidates,
            "rejected_titles": ["Only"],
            "draft_recommendation": None,
        })

        assert "Only" not in result["final_response"]
        assert "couldn't find" in result["final_response"].lower()


class TestMovieNightWorkflowWithRecommendationWriter:
    def test_movies_path_uses_writer_draft_text(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                "1",
                "The Matrix",
                genres=["Sci-Fi", "Action"],
                rating=8.7,
                overview="A hacker uncovers the truth about reality.",
                runtime_minutes=136,
            ),
            _candidate(
                "2",
                "Some Drama",
                genres=["Drama"],
                rating=6.5,
                overview="Unrelated.",
                runtime_minutes=100,
            ),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
        )
        result = workflow.invoke("Recommend a sci-fi movie")

        assert result["route"] == "movies"
        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "The Matrix"
        assert "The Matrix" in result["final_response"]
        assert "Some Drama" not in result["final_response"]

    def test_hybrid_path_also_runs_writer(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of horror films",
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                "10",
                "Get Out",
                genres=["Horror", "Thriller"],
                rating=7.7,
                overview="A visit gone wrong.",
            ),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
        )
        result = workflow.invoke("Horror movies and their history")

        assert result["route"] == "hybrid"
        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "Get Out"
        assert "Get Out" in result["final_response"]

    def test_writer_skipped_when_no_candidates(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
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
            recommendation_writer=stub_recommendation_writer,
        )
        result = workflow.invoke("Recommend nonexistent genre movies")

        assert result["route"] == "movies"
        assert result["candidate_movies"] == []
        assert result["draft_recommendation"] is None
        assert "couldn't find" in result["final_response"].lower()

    def test_rejected_titles_are_respected_by_writer(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate("1", "The Matrix", genres=["sci-fi"], rating=8.7),
            _candidate("2", "Inception", genres=["sci-fi"], rating=8.8),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
        )

        initial_state = {
            "user_message": "Recommend a sci-fi movie",
            "route": None,
            "constraints": None,
            "needs_recommendation": False,
            "rag_query": None,
            "candidate_movies": [],
            "retrieved_contexts": [],
            "draft_recommendation": None,
            "evaluation_result": None,
            "retry_count": 0,
            "rejected_titles": ["Inception"],
            "final_response": None,
            "error": None,
        }
        result = workflow._graph.invoke(initial_state)

        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "The Matrix"
        assert "Inception" not in result["final_response"]

    def test_rag_route_skips_writer(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        mock_recommendation_writer,
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
            recommendation_writer=mock_recommendation_writer,
        )
        result = workflow.invoke("How does this work?")

        assert result["final_response"] == "This app helps you find movies."
        mock_recommendation_writer.write.assert_not_called()
        mock_movie_finder.find_movies.assert_not_called()

    def test_get_response_returns_grounded_text(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        stub_movie_finder,
        stub_recommendation_writer,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["comedy"]),
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
            recommendation_writer=stub_recommendation_writer,
        )

        reply, route, constraints = workflow.get_response("Recommend a comedy")

        assert route == "movies"
        assert constraints.genres == ["comedy"]
        assert reply
        assert reply.strip() != ""


class TestEvaluateNode:
    def test_skips_when_no_draft_and_marks_retries_exhausted(self, mock_evaluator):
        node = create_evaluate_node(mock_evaluator)

        result = node({
            "user_message": "anything",
            "constraints": Constraints(),
            "draft_recommendation": None,
            "rejected_titles": [],
            "retry_count": 0,
        })

        assert result == {"retry_count": MAX_RETRIES}
        mock_evaluator.evaluate.assert_not_called()

    def test_passes_draft_when_evaluator_accepts(self, mock_evaluator):
        movie = _candidate("1", "Great Movie", genres=["Action"])
        draft = DraftRecommendation(
            movie=movie, recommendation_text="A good pick."
        )
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=True, score=0.9, feedback="ok"
        )

        node = create_evaluate_node(mock_evaluator)
        result = node({
            "user_message": "recommend",
            "constraints": Constraints(),
            "draft_recommendation": draft,
            "rejected_titles": [],
            "retry_count": 0,
        })

        assert result["evaluation_result"].passed is True
        assert "retry_count" not in result
        assert "rejected_titles" not in result
        assert "draft_recommendation" not in result

    def test_fails_draft_increments_retry_and_appends_rejected(
        self, mock_evaluator
    ):
        movie = _candidate("1", "Rejected Pick", genres=["Action"])
        draft = DraftRecommendation(
            movie=movie, recommendation_text="Some text."
        )
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=False, score=0.3, feedback="bad"
        )

        node = create_evaluate_node(mock_evaluator)
        result = node({
            "user_message": "recommend",
            "constraints": Constraints(),
            "draft_recommendation": draft,
            "rejected_titles": ["Old Reject"],
            "retry_count": 1,
        })

        assert result["evaluation_result"].passed is False
        assert result["retry_count"] == 2
        assert result["rejected_titles"] == ["Old Reject", "Rejected Pick"]
        assert result["draft_recommendation"] is None

    def test_score_below_threshold_fails_even_if_passed_true(
        self, mock_evaluator
    ):
        movie = _candidate("1", "Weak", genres=["Action"])
        draft = DraftRecommendation(
            movie=movie, recommendation_text="Some text."
        )
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=True, score=0.5, feedback="weak"
        )

        node = create_evaluate_node(mock_evaluator)
        result = node({
            "user_message": "recommend",
            "constraints": Constraints(),
            "draft_recommendation": draft,
            "rejected_titles": [],
            "retry_count": 0,
        })

        assert result["retry_count"] == 1
        assert result["rejected_titles"] == ["Weak"]
        assert result["draft_recommendation"] is None

    def test_does_not_duplicate_rejected_title(self, mock_evaluator):
        movie = _candidate("1", "Already Here", genres=["Action"])
        draft = DraftRecommendation(
            movie=movie, recommendation_text="Some text."
        )
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=False, score=0.2, feedback="bad"
        )

        node = create_evaluate_node(mock_evaluator)
        result = node({
            "user_message": "recommend",
            "constraints": Constraints(),
            "draft_recommendation": draft,
            "rejected_titles": ["Already Here"],
            "retry_count": 0,
        })

        assert result["rejected_titles"] == ["Already Here"]

    def test_forwards_rejected_titles_to_evaluator(self, mock_evaluator):
        movie = _candidate("1", "Clean", genres=["Action"])
        draft = DraftRecommendation(
            movie=movie, recommendation_text="ok"
        )
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=True, score=0.9, feedback="ok"
        )

        node = create_evaluate_node(mock_evaluator)
        node({
            "user_message": "request",
            "constraints": Constraints(genres=["action"]),
            "draft_recommendation": draft,
            "rejected_titles": ["Bad One"],
            "retry_count": 0,
        })

        kwargs = mock_evaluator.evaluate.call_args.kwargs
        assert kwargs["user_message"] == "request"
        assert kwargs["constraints"].genres == ["action"]
        assert kwargs["draft"] is draft
        assert kwargs["rejected_titles"] == ["Bad One"]


class TestRouteAfterEvaluate:
    def test_routes_to_respond_when_draft_survives(self):
        draft = DraftRecommendation(
            movie=_candidate("1", "Passed", genres=["A"]),
            recommendation_text="ok",
        )
        state = {
            "draft_recommendation": draft,
            "evaluation_result": EvaluationResult(
                passed=True, score=0.9, feedback="ok"
            ),
            "retry_count": 0,
        }
        assert route_after_evaluate(state) == "respond"

    def test_routes_to_respond_when_no_evaluation_happened(self):
        state = {
            "draft_recommendation": None,
            "evaluation_result": None,
            "retry_count": 0,
        }
        assert route_after_evaluate(state) == "respond"

    def test_routes_to_writer_when_retry_available(self):
        state = {
            "draft_recommendation": None,
            "evaluation_result": EvaluationResult(
                passed=False, score=0.1, feedback="bad"
            ),
            "retry_count": 1,
        }
        assert route_after_evaluate(state) == "write_recommendation"

    def test_routes_to_respond_when_retries_exhausted(self):
        state = {
            "draft_recommendation": None,
            "evaluation_result": EvaluationResult(
                passed=False, score=0.1, feedback="bad"
            ),
            "retry_count": MAX_RETRIES,
        }
        assert route_after_evaluate(state) == "respond"


class TestRespondNodeWithEvaluation:
    def test_respond_uses_safe_fallback_when_retries_exhausted(
        self, mock_movies_responder, mock_system_responder
    ):
        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Recommend something",
            "route": "movies",
            "constraints": Constraints(),
            "candidate_movies": [
                _candidate("1", "Rejected A"),
                _candidate("2", "Rejected B"),
            ],
            "rejected_titles": ["Rejected A", "Rejected B"],
            "draft_recommendation": None,
            "evaluation_result": EvaluationResult(
                passed=False, score=0.1, feedback="still bad"
            ),
            "retry_count": MAX_RETRIES,
        })

        assert result["final_response"] == RETRY_EXHAUSTED_FALLBACK_MESSAGE

    def test_respond_prefers_draft_over_fallback(
        self, mock_movies_responder, mock_system_responder
    ):
        draft = DraftRecommendation(
            movie=_candidate("1", "Accepted", genres=["Action"]),
            recommendation_text="A solid pick for action fans.",
        )
        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Recommend something",
            "route": "movies",
            "constraints": Constraints(),
            "candidate_movies": [draft.movie],
            "rejected_titles": [],
            "draft_recommendation": draft,
            "evaluation_result": EvaluationResult(
                passed=True, score=0.9, feedback="ok"
            ),
            "retry_count": 0,
        })

        assert result["final_response"] == "A solid pick for action fans."

    def test_respond_no_candidates_still_uses_no_match_message(
        self, mock_movies_responder, mock_system_responder
    ):
        node = create_respond_node(mock_movies_responder, mock_system_responder)
        result = node({
            "user_message": "Recommend something",
            "route": "movies",
            "constraints": Constraints(),
            "candidate_movies": [],
            "rejected_titles": [],
            "draft_recommendation": None,
            "evaluation_result": None,
            "retry_count": 0,
        })

        assert "couldn't find" in result["final_response"].lower()
        assert result["final_response"] != RETRY_EXHAUSTED_FALLBACK_MESSAGE


class TestMovieNightWorkflowWithEvaluator:
    def test_happy_path_passes_on_first_try(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        stub_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                "1",
                "The Matrix",
                genres=["Sci-Fi"],
                rating=8.7,
                overview="A hacker uncovers the truth.",
                runtime_minutes=136,
            ),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=stub_evaluator,
        )
        result = workflow.invoke("Recommend a sci-fi movie")

        assert result["route"] == "movies"
        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "The Matrix"
        assert result["evaluation_result"] is not None
        assert result["evaluation_result"].passed is True
        assert result["retry_count"] == 0
        assert result["rejected_titles"] == []
        assert "The Matrix" in result["final_response"]

    def test_retry_loop_succeeds_after_first_fail(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                "1",
                "Inception",
                genres=["Sci-Fi"],
                rating=8.8,
                overview="Dream heist.",
                runtime_minutes=148,
            ),
            _candidate(
                "2",
                "The Matrix",
                genres=["Sci-Fi"],
                rating=8.7,
                overview="Truth about reality.",
                runtime_minutes=136,
            ),
        ]
        mock_evaluator.evaluate.side_effect = [
            EvaluationResult(
                passed=False, score=0.2, feedback="off-topic"
            ),
            EvaluationResult(
                passed=True, score=0.9, feedback="great"
            ),
        ]

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend a sci-fi movie")

        assert result["retry_count"] == 1
        assert "Inception" in result["rejected_titles"]
        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "The Matrix"
        assert result["evaluation_result"].passed is True
        assert "The Matrix" in result["final_response"]
        assert "Inception" not in result["final_response"]
        assert mock_evaluator.evaluate.call_count == 2

    def test_retry_loop_stops_at_max_retries_and_returns_fallback(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                f"{i}",
                f"Movie {i}",
                genres=["Sci-Fi"],
                rating=7.0 + i * 0.1,
                overview=f"Overview {i}.",
                runtime_minutes=120,
            )
            for i in range(1, 6)
        ]
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=False, score=0.1, feedback="always bad"
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend a sci-fi movie")

        assert result["retry_count"] == MAX_RETRIES
        assert len(result["rejected_titles"]) == MAX_RETRIES
        assert result["draft_recommendation"] is None
        assert result["final_response"] == RETRY_EXHAUSTED_FALLBACK_MESSAGE
        assert mock_evaluator.evaluate.call_count == MAX_RETRIES

    def test_retry_loop_stops_when_writer_runs_out_of_candidates(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                "1",
                "Only Option",
                genres=["Sci-Fi"],
                rating=8.0,
                overview="just this.",
                runtime_minutes=100,
            ),
        ]
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=False, score=0.1, feedback="bad"
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend a sci-fi movie")

        assert result["rejected_titles"] == ["Only Option"]
        assert result["draft_recommendation"] is None
        assert result["final_response"] == RETRY_EXHAUSTED_FALLBACK_MESSAGE
        assert mock_evaluator.evaluate.call_count == 1

    def test_rag_route_skips_evaluator(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="how does it work?",
        )
        mock_system_responder.respond.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("how does this work?")

        assert result["final_response"] == "This app helps you find movies."
        mock_evaluator.evaluate.assert_not_called()
        mock_movie_finder.find_movies.assert_not_called()

    def test_clarification_skips_evaluator(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(),
            needs_clarification=True,
            clarification_question="What genre?",
            needs_recommendation=False,
            rag_query=None,
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("help")

        assert result["route"] == "clarification"
        assert result["final_response"] == "What genre?"
        mock_evaluator.evaluate.assert_not_called()

    def test_no_candidates_skips_evaluator(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
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
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend nonexistent genre movies")

        assert result["candidate_movies"] == []
        assert result["draft_recommendation"] is None
        mock_evaluator.evaluate.assert_not_called()
        assert "couldn't find" in result["final_response"].lower()

    def test_get_response_returns_fallback_when_retries_exhausted(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["sci-fi"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            _candidate(
                f"{i}",
                f"Movie {i}",
                genres=["Sci-Fi"],
                rating=7.0,
                runtime_minutes=100,
            )
            for i in range(1, 5)
        ]
        mock_evaluator.evaluate.return_value = EvaluationResult(
            passed=False, score=0.1, feedback="bad"
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
        )

        reply, route, constraints = workflow.get_response("sci-fi please")

        assert route == "movies"
        assert reply == RETRY_EXHAUSTED_FALLBACK_MESSAGE
