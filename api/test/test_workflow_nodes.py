"""Unit tests for workflow node functions."""

from app.llm.state import MAX_RETRIES
from app.llm.workflow import (
    RETRY_EXHAUSTED_FALLBACK_MESSAGE,
    create_evaluate_node,
    create_find_movies_node,
    create_input_orchestrate_node,
    create_orchestrate_node,
    create_rag_respond_node,
    create_rag_retrieve_node,
    create_respond_node,
    create_write_recommendation_node,
)
from app.schemas.domain import (
    DraftRecommendation,
    EvaluationResult,
    MovieResult,
    RetrievedContext,
)
from app.schemas.orchestrator import Constraints, InputDecision, OrchestratorDecision

from conftest import make_movie


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
        movie = make_movie("1", "The Matrix", genres=["Sci-Fi"], rating=8.7)
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
            "candidate_movies": [make_movie("1", "Any")],
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
            "candidate_movies": [make_movie("1", "Only")],
            "rejected_titles": ["Only"],
        })

        assert result == {"draft_recommendation": None}


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
        movie = make_movie("1", "Great Movie", genres=["Action"])
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
        movie = make_movie("1", "Rejected Pick", genres=["Action"])
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
        movie = make_movie("1", "Weak", genres=["Action"])
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
        movie = make_movie("1", "Already Here", genres=["Action"])
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
        movie = make_movie("1", "Clean", genres=["Action"])
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


class TestRAGRetrieveNode:
    def test_rag_retrieve_uses_rag_query(self, mock_rag_retriever):
        mock_rag_retriever.retrieve.return_value = [
            RetrievedContext(
                content="System overview content",
                source="rag",
                relevance_score=0.9,
                metadata={"title": "System Overview"},
            ),
        ]

        node = create_rag_retrieve_node(mock_rag_retriever)
        result = node({
            "user_message": "How does this work?",
            "rag_query": "system overview",
        })

        mock_rag_retriever.retrieve.assert_called_once_with("system overview")
        assert len(result["retrieved_contexts"]) == 1
        assert result["retrieved_contexts"][0].source == "rag"

    def test_rag_retrieve_falls_back_to_user_message(self, mock_rag_retriever):
        mock_rag_retriever.retrieve.return_value = []

        node = create_rag_retrieve_node(mock_rag_retriever)
        result = node({
            "user_message": "What is this app?",
            "rag_query": None,
        })

        mock_rag_retriever.retrieve.assert_called_once_with("What is this app?")
        assert result["retrieved_contexts"] == []


class TestRAGRespondNode:
    def test_rag_respond_generates_answer(self, mock_rag_agent):
        mock_rag_agent.answer.return_value = "This is the RAG-grounded answer."

        contexts = [
            RetrievedContext(
                content="Documentation content",
                source="rag",
                metadata={"title": "Docs"},
            ),
        ]

        node = create_rag_respond_node(mock_rag_agent)
        result = node({
            "user_message": "How does this work?",
            "rag_query": "system functionality",
            "retrieved_contexts": contexts,
        })

        mock_rag_agent.answer.assert_called_once_with(
            query="system functionality",
            contexts=contexts,
        )
        assert result["final_response"] == "This is the RAG-grounded answer."

    def test_rag_respond_uses_user_message_when_no_rag_query(self, mock_rag_agent):
        mock_rag_agent.answer.return_value = "Answer"

        node = create_rag_respond_node(mock_rag_agent)
        result = node({
            "user_message": "What is this?",
            "rag_query": None,
            "retrieved_contexts": [],
        })

        mock_rag_agent.answer.assert_called_once_with(
            query="What is this?",
            contexts=[],
        )


class TestRespondNodeWithDraft:
    def test_respond_uses_draft_text_when_available(
        self, mock_movies_responder, mock_system_responder
    ):
        candidates = [make_movie("1", "Fallback Title", genres=["Action"])]
        draft = DraftRecommendation(
            movie=make_movie(
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
        candidates = [make_movie("1", "Fallback Title", genres=["Action"])]

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
            make_movie("1", "Rejected Pick", genres=["Action"]),
            make_movie("2", "Safe Pick", genres=["Action"]),
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
            make_movie("1", "Too Long", genres=["Action"], runtime_minutes=200),
            make_movie("2", "Just Right", genres=["Action"], runtime_minutes=90),
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
        candidates = [make_movie("1", "Only", genres=["Action"])]

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
                make_movie("1", "Rejected A"),
                make_movie("2", "Rejected B"),
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
            movie=make_movie("1", "Accepted", genres=["Action"]),
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
