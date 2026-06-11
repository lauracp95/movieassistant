"""Unit tests for workflow routing functions."""

from langgraph.graph import END

from app.workflow.state import MAX_RETRIES
from app.workflow import (
    route_after_evaluate,
    route_after_orchestrate,
)
from app.schemas.domain import DraftRecommendation, EvaluationResult

from conftest import make_movie


class TestRouteAfterOrchestrate:
    def test_routes_to_find_movies_for_movies_route(self):
        assert route_after_orchestrate({"route": "movies"}) == "find_movies"

    def test_routes_to_find_movies_for_hybrid_route(self):
        assert route_after_orchestrate({"route": "hybrid"}) == "find_movies"

    def test_routes_to_rag_retrieve_for_rag_route(self):
        assert route_after_orchestrate({"route": "rag"}) == "rag_retrieve"

    def test_routes_to_end_for_unknown_route(self):
        assert route_after_orchestrate({"route": "unknown"}) == END

    def test_routes_to_end_for_none_route(self):
        assert route_after_orchestrate({"route": None}) == END

    def test_routes_to_end_for_clarification(self):
        assert route_after_orchestrate({"route": "clarification"}) == END


class TestRouteAfterEvaluate:
    def test_routes_to_end_when_draft_survives(self):
        draft = DraftRecommendation(
            movie=make_movie("1", "Passed", genres=["A"]),
            recommendation_text="ok",
        )
        state = {
            "draft_recommendation": draft,
            "evaluation_result": EvaluationResult(
                passed=True, score=0.9, feedback="ok"
            ),
            "retry_count": 0,
        }
        assert route_after_evaluate(state) == END

    def test_routes_to_end_when_no_evaluation_happened(self):
        state = {
            "draft_recommendation": None,
            "evaluation_result": None,
            "retry_count": 0,
        }
        assert route_after_evaluate(state) == END

    def test_routes_to_writer_when_retry_available(self):
        state = {
            "draft_recommendation": None,
            "evaluation_result": EvaluationResult(
                passed=False, score=0.1, feedback="bad"
            ),
            "retry_count": 1,
        }
        assert route_after_evaluate(state) == "write_recommendation"

    def test_routes_to_end_when_retries_exhausted(self):
        state = {
            "draft_recommendation": None,
            "evaluation_result": EvaluationResult(
                passed=False, score=0.1, feedback="bad"
            ),
            "retry_count": MAX_RETRIES,
        }
        assert route_after_evaluate(state) == END
