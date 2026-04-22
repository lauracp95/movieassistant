"""Integration tests for MovieNightWorkflow basic functionality."""

import pytest

from app.llm.workflow import MovieNightWorkflow
from app.schemas.domain import MovieResult
from app.schemas.orchestrator import Constraints, InputDecision, OrchestratorDecision

from conftest import make_movie


class TestMovieNightWorkflowBasic:
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
            make_movie(
                "1",
                "The Matrix",
                genres=["Sci-Fi", "Action"],
                rating=8.7,
                overview="A hacker uncovers the truth about reality.",
                runtime_minutes=136,
            ),
            make_movie(
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
            make_movie(
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
            make_movie("1", "The Matrix", genres=["sci-fi"], rating=8.7),
            make_movie("2", "Inception", genres=["sci-fi"], rating=8.8),
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
