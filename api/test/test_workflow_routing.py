"""Unit tests for workflow routing functions."""

from langgraph.graph import END

from app.llm.state import MAX_RETRIES
from app.llm.workflow import (
    route_after_evaluate,
    route_after_orchestrate,
    should_respond,
)
from app.schemas.domain import DraftRecommendation, EvaluationResult

from conftest import make_movie


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
        assert should_respond({"route": "clarification"}) == END


class TestRouteAfterOrchestrate:
    def test_routes_to_find_movies_for_movies_route(self):
        assert route_after_orchestrate({"route": "movies"}) == "find_movies"

    def test_routes_to_find_movies_for_hybrid_route(self):
        assert route_after_orchestrate({"route": "hybrid"}) == "find_movies"

    def test_routes_to_respond_for_rag_route(self):
        assert route_after_orchestrate({"route": "rag"}) == "respond"

    def test_routes_to_end_for_clarification(self):
        assert route_after_orchestrate({"route": "clarification"}) == END


class TestRouteAfterEvaluate:
    def test_routes_to_respond_when_draft_survives(self):
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
