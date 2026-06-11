"""Integration tests for MovieNightWorkflow basic functionality."""

import pytest

from app.workflow import MovieNightWorkflow
from app.schemas.domain import DraftRecommendation, MovieResult
from app.schemas.input import Constraints, InputDecision

from conftest import make_movie


class TestMovieNightWorkflowWithInputAgent:
    def test_workflow_movies_with_input_agent(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert result["constraints"].genres == ["action"]
        assert result["needs_recommendation"] is True
        assert len(result["candidate_movies"]) > 0

    def test_workflow_rag_with_input_agent(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does the app work?",
        )
        mock_rag_retriever.retrieve.return_value = []
        mock_rag_agent.answer.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("How do you work?")

        assert result["route"] == "rag"
        assert result["final_response"] == "This app helps you find movies."
        assert result["needs_recommendation"] is False
        assert result["rag_query"] == "How does the app work?"

    def test_workflow_hybrid_with_input_agent(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of Halloween horror films",
        )
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Horror movies for Halloween and their history")

        assert result["route"] == "hybrid"
        assert result["constraints"].genres == ["horror"]
        assert result["needs_recommendation"] is True
        assert result["rag_query"] == "History of Halloween horror films"
        assert len(result["candidate_movies"]) > 0

    def test_workflow_clarification_with_input_agent(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
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
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("something")

        assert result["route"] == "clarification"
        assert result["final_response"] == "What mood are you in?"

    def test_workflow_requires_input_agent(
        self,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        with pytest.raises(ValueError, match="input_agent must be provided"):
            MovieNightWorkflow(
                input_agent=None,
                movie_finder=in_memory_movie_finder,
                rag_retriever=mock_rag_retriever,
                rag_agent=mock_rag_agent,
                recommendation_writer=recommendation_writer,
                evaluator=mock_evaluator,
            )

    def test_get_response_with_input_agent(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does the app work?",
        )
        mock_rag_retriever.retrieve.return_value = []
        mock_rag_agent.answer.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        reply, route, constraints = workflow.get_response("How does this work?")

        assert reply == "This app helps you find movies."
        assert route == "rag"


class TestMovieNightWorkflowWithMovieFinder:
    def test_workflow_movies_with_finder(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        mock_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        action_hero = MovieResult(
            id="test-1",
            title="Action Hero",
            year=2023,
            genres=["Action"],
            overview="An action-packed adventure.",
            rating=8.5,
            source="test",
        )
        mock_movie_finder.find_movies.return_value = [action_hero]
        mock_rag_retriever.retrieve.return_value = []
        mock_recommendation_writer.write.return_value = DraftRecommendation(
            movie=action_hero,
            recommendation_text="Action Hero is a thrilling pick.",
        )

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=mock_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend action movies")

        assert result["route"] == "movies"
        assert len(result["candidate_movies"]) == 1
        assert result["candidate_movies"][0].title == "Action Hero"
        assert "Action Hero" in result["final_response"]
        mock_movie_finder.find_movies.assert_called_once()

    def test_workflow_hybrid_with_finder(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        mock_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of horror films",
        )
        scary_movie = MovieResult(
            id="test-2",
            title="Scary Movie",
            year=2020,
            genres=["Horror"],
            overview="A terrifying experience.",
            rating=7.0,
            source="test",
        )
        mock_movie_finder.find_movies.return_value = [scary_movie]
        mock_rag_retriever.retrieve.return_value = []
        mock_recommendation_writer.write.return_value = DraftRecommendation(
            movie=scary_movie,
            recommendation_text="Scary Movie will keep you on edge.",
        )

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=mock_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Horror movies and their history")

        assert result["route"] == "hybrid"
        assert len(result["candidate_movies"]) == 1
        assert "Scary Movie" in result["final_response"]
        mock_movie_finder.find_movies.assert_called_once()

    def test_workflow_rag_skips_finder(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does this work?",
        )
        mock_rag_retriever.retrieve.return_value = []
        mock_rag_agent.answer.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("How does this work?")

        assert result["route"] == "rag"
        assert result["final_response"] == "This app helps you find movies."
        mock_movie_finder.find_movies.assert_not_called()

    def test_workflow_with_in_memory_finder_integration(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Horror movies please")

        assert result["route"] == "movies"
        assert len(result["candidate_movies"]) > 0
        for movie in result["candidate_movies"]:
            assert any("horror" in g.lower() for g in movie.genres)

    def test_workflow_empty_finder_results(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
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
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend nonexistent genre movies")

        assert result["route"] == "movies"
        assert result["candidate_movies"] == []
        assert "couldn't find" in result["final_response"].lower()

    def test_workflow_clarification_skips_finder(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        mock_evaluator,
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
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("recommend something")

        assert result["route"] == "clarification"
        assert result["final_response"] == "What genre do you prefer?"
        mock_movie_finder.find_movies.assert_not_called()


class TestMovieNightWorkflowWithRecommendationWriter:
    def test_movies_path_uses_writer_draft_text(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        evaluator,
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
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=evaluator,
        )
        result = workflow.invoke("Recommend a sci-fi movie")

        assert result["route"] == "movies"
        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "The Matrix"
        assert (
            result["final_response"]
            == result["draft_recommendation"].recommendation_text
        )
        assert "Some Drama" not in result["final_response"]

    def test_hybrid_path_also_runs_writer(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        evaluator,
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
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=evaluator,
        )
        result = workflow.invoke("Horror movies and their history")

        assert result["route"] == "hybrid"
        assert result["draft_recommendation"] is not None
        assert result["draft_recommendation"].movie.title == "Get Out"
        assert (
            result["final_response"]
            == result["draft_recommendation"].recommendation_text
        )

    def test_writer_skipped_when_no_candidates(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
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
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("Recommend nonexistent genre movies")

        assert result["route"] == "movies"
        assert result["candidate_movies"] == []
        assert result["draft_recommendation"] is None
        assert "couldn't find" in result["final_response"].lower()

    def test_rejected_titles_are_respected_by_writer(
        self,
        mock_input_agent,
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        evaluator,
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
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=evaluator,
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
        mock_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        mock_recommendation_writer,
        mock_evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does this work?",
        )
        mock_rag_retriever.retrieve.return_value = []
        mock_rag_agent.answer.return_value = "This app helps you find movies."

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=mock_recommendation_writer,
            evaluator=mock_evaluator,
        )
        result = workflow.invoke("How does this work?")

        assert result["final_response"] == "This app helps you find movies."
        mock_recommendation_writer.write.assert_not_called()
        mock_movie_finder.find_movies.assert_not_called()

    def test_get_response_returns_grounded_text(
        self,
        mock_input_agent,
        in_memory_movie_finder,
        mock_rag_retriever,
        mock_rag_agent,
        recommendation_writer,
        evaluator,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["comedy"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_rag_retriever.retrieve.return_value = []

        workflow = MovieNightWorkflow(
            input_agent=mock_input_agent,
            movie_finder=in_memory_movie_finder,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=evaluator,
        )

        reply, route, constraints = workflow.get_response("Recommend a comedy")

        assert route == "movies"
        assert constraints.genres == ["comedy"]
        assert reply
        assert reply.strip() != ""
