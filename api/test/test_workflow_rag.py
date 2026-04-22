"""Integration tests for MovieNightWorkflow RAG functionality."""

from app.llm.workflow import MovieNightWorkflow
from app.schemas.domain import RetrievedContext
from app.schemas.orchestrator import Constraints, InputDecision

from conftest import make_movie


class TestRAGWorkflowIntegration:
    def test_rag_route_uses_rag_pipeline(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
        mock_rag_retriever,
        mock_rag_agent,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="how does the system work?",
        )
        mock_rag_retriever.retrieve.return_value = [
            RetrievedContext(
                content="The system uses TMDB for movie data.",
                source="rag",
                relevance_score=0.9,
                metadata={"title": "System Overview"},
            ),
        ]
        mock_rag_agent.answer.return_value = "The Movie Night Assistant uses TMDB for movie data."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
        )
        result = workflow.invoke("How does this system work?")

        assert result["route"] == "rag"
        assert result["final_response"] == "The Movie Night Assistant uses TMDB for movie data."
        assert len(result["retrieved_contexts"]) == 1
        mock_rag_retriever.retrieve.assert_called_once()
        mock_rag_agent.answer.assert_called_once()
        mock_movie_finder.find_movies.assert_not_called()
        mock_evaluator.evaluate.assert_not_called()

    def test_hybrid_route_uses_both_movies_and_rag(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        stub_evaluator,
        mock_rag_retriever,
        stub_rag_agent,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["comedy"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="what makes a good comedy?",
        )
        mock_movie_finder.find_movies.return_value = [
            make_movie(
                "1",
                "Funny Movie",
                genres=["Comedy"],
                rating=8.0,
                overview="A funny movie.",
                runtime_minutes=100,
            ),
        ]
        mock_rag_retriever.retrieve.return_value = [
            RetrievedContext(
                content="Comedy recommendations focus on humor.",
                source="rag",
                metadata={"title": "Recommendation Rules"},
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
            rag_retriever=mock_rag_retriever,
            rag_agent=stub_rag_agent,
        )
        result = workflow.invoke("Recommend a comedy and explain why it's funny")

        assert result["route"] == "hybrid"
        assert result["candidate_movies"] is not None
        assert len(result["candidate_movies"]) > 0
        assert len(result["retrieved_contexts"]) > 0
        mock_movie_finder.find_movies.assert_called_once()
        mock_rag_retriever.retrieve.assert_called_once()

    def test_movies_route_with_rag_enabled_skips_rag(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        stub_evaluator,
        mock_rag_retriever,
        mock_rag_agent,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["action"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )
        mock_movie_finder.find_movies.return_value = [
            make_movie(
                "1",
                "Action Hero",
                genres=["Action"],
                rating=7.5,
                overview="An action movie.",
                runtime_minutes=120,
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
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
        )
        result = workflow.invoke("Recommend an action movie")

        assert result["route"] == "movies"
        mock_movie_finder.find_movies.assert_called_once()
        mock_rag_retriever.retrieve.assert_not_called()
        mock_rag_agent.answer.assert_not_called()

    def test_rag_route_without_rag_components_falls_back_to_system_responder(
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
        mock_system_responder.respond.return_value = "Fallback system response."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
            rag_retriever=None,
            rag_agent=None,
        )
        result = workflow.invoke("how does this work?")

        assert result["final_response"] == "Fallback system response."
        mock_system_responder.respond.assert_called_once()

    def test_rag_route_populates_retrieved_contexts_in_state(
        self,
        mock_input_agent,
        mock_movies_responder,
        mock_system_responder,
        mock_movie_finder,
        stub_recommendation_writer,
        mock_evaluator,
        mock_rag_retriever,
        mock_rag_agent,
    ):
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="what are the limitations?",
        )
        expected_contexts = [
            RetrievedContext(
                content="The system has no memory.",
                source="rag",
                relevance_score=0.8,
                metadata={"title": "Known Limitations"},
            ),
            RetrievedContext(
                content="No streaming availability info.",
                source="rag",
                relevance_score=0.7,
                metadata={"title": "Known Limitations"},
            ),
        ]
        mock_rag_retriever.retrieve.return_value = expected_contexts
        mock_rag_agent.answer.return_value = "Here are the limitations..."

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=mock_movie_finder,
            recommendation_writer=stub_recommendation_writer,
            evaluator=mock_evaluator,
            rag_retriever=mock_rag_retriever,
            rag_agent=mock_rag_agent,
        )
        result = workflow.invoke("What are the limitations?")

        assert result["retrieved_contexts"] == expected_contexts
        assert len(result["retrieved_contexts"]) == 2
