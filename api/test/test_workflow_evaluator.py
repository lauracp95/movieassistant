"""Integration tests for MovieNightWorkflow evaluator functionality."""

from app.llm.state import MAX_RETRIES
from app.llm.workflow import RETRY_EXHAUSTED_FALLBACK_MESSAGE, MovieNightWorkflow
from app.schemas.domain import EvaluationResult
from app.schemas.orchestrator import Constraints, InputDecision

from conftest import make_movie


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
            make_movie(
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
            make_movie(
                "1",
                "Inception",
                genres=["Sci-Fi"],
                rating=8.8,
                overview="Dream heist.",
                runtime_minutes=148,
            ),
            make_movie(
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
            make_movie(
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
            make_movie(
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
            make_movie(
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
