"""Unit tests for domain models."""

import pytest
from pydantic import ValidationError

from app.schemas.domain import (
    DraftRecommendation,
    EvaluationResult,
    MovieResult,
    RetrievedContext,
    RouteDecision,
)


class TestMovieResult:
    def test_minimal_movie_result(self):
        movie = MovieResult(id="123", title="Test Movie")
        assert movie.id == "123"
        assert movie.title == "Test Movie"
        assert movie.year is None
        assert movie.genres == []
        assert movie.runtime_minutes is None
        assert movie.source == "unknown"

    def test_full_movie_result(self):
        movie = MovieResult(
            id="tmdb-550",
            title="Fight Club",
            year=1999,
            genres=["Drama", "Thriller"],
            runtime_minutes=139,
            overview="An insomniac office worker...",
            rating=8.4,
            poster_url="https://image.tmdb.org/t/p/w500/poster.jpg",
            source="tmdb",
        )
        assert movie.id == "tmdb-550"
        assert movie.title == "Fight Club"
        assert movie.year == 1999
        assert movie.genres == ["Drama", "Thriller"]
        assert movie.runtime_minutes == 139
        assert movie.rating == 8.4
        assert movie.source == "tmdb"

    def test_rating_validation_bounds(self):
        MovieResult(id="1", title="Test", rating=0.0)
        MovieResult(id="1", title="Test", rating=10.0)

        with pytest.raises(ValidationError):
            MovieResult(id="1", title="Test", rating=-0.1)

        with pytest.raises(ValidationError):
            MovieResult(id="1", title="Test", rating=10.1)

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            MovieResult(title="Test")

        with pytest.raises(ValidationError):
            MovieResult(id="123")


class TestDraftRecommendation:
    def test_draft_recommendation(self):
        movie = MovieResult(id="123", title="Test Movie")
        draft = DraftRecommendation(
            movie=movie,
            recommendation_text="This is a great movie because...",
            reasoning="Selected based on genre match",
        )
        assert draft.movie.title == "Test Movie"
        assert "great movie" in draft.recommendation_text
        assert draft.reasoning is not None

    def test_draft_without_reasoning(self):
        movie = MovieResult(id="123", title="Test Movie")
        draft = DraftRecommendation(
            movie=movie,
            recommendation_text="Recommended for you",
        )
        assert draft.reasoning is None


class TestEvaluationResult:
    def test_passing_evaluation(self):
        result = EvaluationResult(
            passed=True,
            score=0.85,
            feedback="Excellent recommendation that meets all criteria.",
        )
        assert result.passed is True
        assert result.score == 0.85
        assert result.constraint_violations == []
        assert result.improvement_suggestions == []

    def test_failing_evaluation(self):
        result = EvaluationResult(
            passed=False,
            score=0.4,
            feedback="Does not meet runtime constraint.",
            constraint_violations=["runtime exceeds max_runtime_minutes"],
            improvement_suggestions=["Find a shorter movie"],
        )
        assert result.passed is False
        assert result.score == 0.4
        assert len(result.constraint_violations) == 1
        assert len(result.improvement_suggestions) == 1

    def test_score_validation_bounds(self):
        EvaluationResult(passed=True, score=0.0, feedback="Min score")
        EvaluationResult(passed=True, score=1.0, feedback="Max score")

        with pytest.raises(ValidationError):
            EvaluationResult(passed=True, score=-0.1, feedback="Invalid")

        with pytest.raises(ValidationError):
            EvaluationResult(passed=True, score=1.1, feedback="Invalid")


class TestRetrievedContext:
    def test_retrieved_context(self):
        ctx = RetrievedContext(
            content="This movie won 3 Academy Awards...",
            source="rag",
            relevance_score=0.92,
            metadata={"chunk_id": "42", "document": "awards.txt"},
        )
        assert "Academy Awards" in ctx.content
        assert ctx.source == "rag"
        assert ctx.relevance_score == 0.92
        assert ctx.metadata["chunk_id"] == "42"

    def test_minimal_context(self):
        ctx = RetrievedContext(content="Some content", source="web")
        assert ctx.relevance_score is None
        assert ctx.metadata == {}


class TestRouteDecision:
    def test_movies_route(self):
        decision = RouteDecision(route="movies", confidence=0.95)
        assert decision.route == "movies"
        assert decision.confidence == 0.95
        assert decision.clarification_needed is False

    def test_system_route(self):
        decision = RouteDecision(route="system")
        assert decision.route == "system"
        assert decision.confidence is None

    def test_clarification_route(self):
        decision = RouteDecision(
            route="clarification",
            clarification_needed=True,
            clarification_question="What genre are you interested in?",
        )
        assert decision.route == "clarification"
        assert decision.clarification_needed is True
        assert decision.clarification_question is not None

    def test_invalid_route(self):
        with pytest.raises(ValidationError):
            RouteDecision(route="invalid_route")
