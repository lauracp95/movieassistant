"""Unit tests for the RecommendationWriterAgent (Phase 4)."""

from unittest.mock import MagicMock

import pytest
from langchain_openai import AzureChatOpenAI

from app.llm.recommendation_agent import (
    LLMRecommendationWriterAgent,
    StubRecommendationWriterAgent,
    build_reasoning,
    filter_candidates,
    prioritize_candidates,
    select_best_candidate,
)
from app.schemas.domain import DraftRecommendation, MovieResult
from app.schemas.orchestrator import Constraints


def _movie(
    id_: str,
    title: str,
    genres: list[str] | None = None,
    runtime_minutes: int | None = None,
    rating: float | None = None,
    year: int | None = None,
    overview: str | None = None,
) -> MovieResult:
    return MovieResult(
        id=id_,
        title=title,
        genres=genres or [],
        runtime_minutes=runtime_minutes,
        rating=rating,
        year=year,
        overview=overview,
        source="test",
    )


class TestFilterCandidates:
    def test_empty_candidates_returns_empty(self):
        assert filter_candidates([], Constraints()) == []

    def test_without_constraints_returns_all(self):
        movies = [
            _movie("1", "A", runtime_minutes=100),
            _movie("2", "B", runtime_minutes=200),
        ]
        assert filter_candidates(movies, Constraints()) == movies

    def test_enforces_max_runtime(self):
        movies = [
            _movie("1", "Short", runtime_minutes=90),
            _movie("2", "Long", runtime_minutes=180),
        ]
        result = filter_candidates(movies, Constraints(max_runtime_minutes=120))
        assert [m.title for m in result] == ["Short"]

    def test_enforces_min_runtime(self):
        movies = [
            _movie("1", "Short", runtime_minutes=80),
            _movie("2", "Long", runtime_minutes=160),
        ]
        result = filter_candidates(movies, Constraints(min_runtime_minutes=120))
        assert [m.title for m in result] == ["Long"]

    def test_runtime_unknown_is_kept(self):
        movies = [_movie("1", "Unknown", runtime_minutes=None)]
        result = filter_candidates(movies, Constraints(max_runtime_minutes=90))
        assert result == movies

    def test_excludes_rejected_titles_case_insensitive(self):
        movies = [
            _movie("1", "The Matrix"),
            _movie("2", "Inception"),
        ]
        result = filter_candidates(movies, Constraints(), rejected_titles=["THE matrix"])
        assert [m.title for m in result] == ["Inception"]

    def test_rejected_titles_none_is_safe(self):
        movies = [_movie("1", "A")]
        assert filter_candidates(movies, Constraints(), rejected_titles=None) == movies

    def test_genre_mismatch_is_not_hard_filtered(self):
        movies = [_movie("1", "Horror Only", genres=["horror"])]
        result = filter_candidates(movies, Constraints(genres=["comedy"]))
        assert result == movies


class TestPrioritizeCandidates:
    def test_sorts_by_genre_overlap_first(self):
        movies = [
            _movie("1", "One Match", genres=["comedy"], rating=6.0),
            _movie("2", "Two Match", genres=["comedy", "action"], rating=5.0),
        ]
        result = prioritize_candidates(
            movies, Constraints(genres=["comedy", "action"])
        )
        assert [m.title for m in result] == ["Two Match", "One Match"]

    def test_tiebreak_by_rating(self):
        movies = [
            _movie("1", "Low", genres=["comedy"], rating=6.0),
            _movie("2", "High", genres=["comedy"], rating=8.5),
        ]
        result = prioritize_candidates(movies, Constraints(genres=["comedy"]))
        assert [m.title for m in result] == ["High", "Low"]

    def test_none_rating_treated_as_zero(self):
        movies = [
            _movie("1", "Rated", genres=["comedy"], rating=4.0),
            _movie("2", "Unrated", genres=["comedy"], rating=None),
        ]
        result = prioritize_candidates(movies, Constraints(genres=["comedy"]))
        assert result[0].title == "Rated"

    def test_no_constraint_genres_still_uses_rating(self):
        movies = [
            _movie("1", "Low", rating=5.0),
            _movie("2", "High", rating=9.0),
        ]
        result = prioritize_candidates(movies, Constraints())
        assert [m.title for m in result] == ["High", "Low"]

    def test_does_not_mutate_input(self):
        movies = [
            _movie("1", "A", rating=5.0),
            _movie("2", "B", rating=8.0),
        ]
        original_order = [m.title for m in movies]
        prioritize_candidates(movies, Constraints())
        assert [m.title for m in movies] == original_order


class TestSelectBestCandidate:
    def test_returns_none_on_empty(self):
        assert select_best_candidate([], Constraints()) is None

    def test_returns_none_when_all_filtered(self):
        movies = [_movie("1", "Long", runtime_minutes=200)]
        result = select_best_candidate(
            movies, Constraints(max_runtime_minutes=100)
        )
        assert result is None

    def test_returns_best_by_genre_then_rating(self):
        movies = [
            _movie("1", "Off Topic", genres=["drama"], rating=9.5),
            _movie("2", "On Topic Weak", genres=["comedy"], rating=6.0),
            _movie("3", "On Topic Strong", genres=["comedy"], rating=8.0),
        ]
        result = select_best_candidate(movies, Constraints(genres=["comedy"]))
        assert result is not None
        assert result.title == "On Topic Strong"

    def test_respects_rejected_titles(self):
        movies = [
            _movie("1", "Best", genres=["comedy"], rating=9.0),
            _movie("2", "Second", genres=["comedy"], rating=8.0),
        ]
        result = select_best_candidate(
            movies, Constraints(genres=["comedy"]), rejected_titles=["Best"]
        )
        assert result is not None
        assert result.title == "Second"


class TestBuildReasoning:
    def test_mentions_matched_genres(self):
        movie = _movie("1", "A", genres=["comedy", "romance"])
        reasoning = build_reasoning(movie, Constraints(genres=["comedy"]))
        assert "comedy" in reasoning

    def test_mentions_rating(self):
        movie = _movie("1", "A", rating=8.4)
        reasoning = build_reasoning(movie, Constraints())
        assert "8.4" in reasoning

    def test_safe_when_nothing_to_say(self):
        movie = _movie("1", "A")
        reasoning = build_reasoning(movie, Constraints())
        assert isinstance(reasoning, str) and reasoning


class TestStubRecommendationWriterAgent:
    def test_returns_none_on_empty_candidates(self):
        writer = StubRecommendationWriterAgent()
        result = writer.write(
            user_message="Any movie",
            constraints=Constraints(),
            candidates=[],
        )
        assert result is None

    def test_returns_none_when_all_rejected(self):
        writer = StubRecommendationWriterAgent()
        movie = _movie("1", "Only", genres=["comedy"])
        result = writer.write(
            user_message="Comedy please",
            constraints=Constraints(genres=["comedy"]),
            candidates=[movie],
            rejected_titles=["Only"],
        )
        assert result is None

    def test_produces_valid_draft(self):
        writer = StubRecommendationWriterAgent()
        movie = _movie(
            "1",
            "The Matrix",
            genres=["Sci-Fi", "Action"],
            runtime_minutes=136,
            rating=8.7,
            year=1999,
            overview="A hacker discovers the truth.",
        )
        result = writer.write(
            user_message="Give me sci-fi",
            constraints=Constraints(genres=["sci-fi"]),
            candidates=[movie],
        )
        assert isinstance(result, DraftRecommendation)
        assert result.movie.title == "The Matrix"
        assert "Matrix" in result.recommendation_text
        assert result.reasoning is not None

    def test_text_grounded_in_overview(self):
        writer = StubRecommendationWriterAgent()
        movie = _movie(
            "1",
            "My Movie",
            genres=["comedy"],
            overview="A very specific plot fingerprint here.",
        )
        result = writer.write(
            user_message="Comedy",
            constraints=Constraints(genres=["comedy"]),
            candidates=[movie],
        )
        assert result is not None
        assert "specific plot fingerprint" in result.recommendation_text

    def test_selects_best_candidate_deterministically(self):
        writer = StubRecommendationWriterAgent()
        worse = _movie("1", "Worse", genres=["comedy"], rating=5.0)
        better = _movie("2", "Better", genres=["comedy"], rating=9.0)
        result = writer.write(
            user_message="Funny movie",
            constraints=Constraints(genres=["comedy"]),
            candidates=[worse, better],
        )
        assert result is not None
        assert result.movie.title == "Better"


class TestLLMRecommendationWriterAgent:
    def test_returns_none_on_empty_candidates(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        writer = LLMRecommendationWriterAgent(llm)
        result = writer.write(
            user_message="anything",
            constraints=Constraints(),
            candidates=[],
        )
        assert result is None
        llm.invoke.assert_not_called()

    def test_calls_llm_for_text_and_grounds_on_selected_movie(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        llm.invoke.return_value = MagicMock(content="Try The Matrix; it fits perfectly.")

        movies = [
            _movie("1", "The Matrix", genres=["Sci-Fi"], rating=8.7,
                   overview="A hacker discovers reality."),
            _movie("2", "Offtopic", genres=["drama"], rating=5.0),
        ]

        writer = LLMRecommendationWriterAgent(llm)
        result = writer.write(
            user_message="Recommend a sci-fi",
            constraints=Constraints(genres=["sci-fi"]),
            candidates=movies,
        )

        assert isinstance(result, DraftRecommendation)
        assert result.movie.title == "The Matrix"
        assert result.recommendation_text == "Try The Matrix; it fits perfectly."
        llm.invoke.assert_called_once()

    def test_falls_back_to_deterministic_text_on_llm_failure(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        llm.invoke.side_effect = RuntimeError("boom")

        movie = _movie(
            "1",
            "Inception",
            genres=["Sci-Fi"],
            rating=8.8,
            overview="Dreams within dreams.",
        )

        writer = LLMRecommendationWriterAgent(llm)
        result = writer.write(
            user_message="Sci-fi please",
            constraints=Constraints(genres=["sci-fi"]),
            candidates=[movie],
        )

        assert isinstance(result, DraftRecommendation)
        assert result.movie.title == "Inception"
        assert "Inception" in result.recommendation_text

    def test_falls_back_on_empty_llm_text(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        llm.invoke.return_value = MagicMock(content="   ")

        movie = _movie("1", "Gravity", genres=["Drama"], rating=7.7)

        writer = LLMRecommendationWriterAgent(llm)
        result = writer.write(
            user_message="Something dramatic",
            constraints=Constraints(genres=["drama"]),
            candidates=[movie],
        )

        assert result is not None
        assert "Gravity" in result.recommendation_text

    def test_honors_rejected_titles(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        llm.invoke.return_value = MagicMock(content="Great pick!")

        movies = [
            _movie("1", "The Matrix", genres=["sci-fi"], rating=8.7),
            _movie("2", "Inception", genres=["sci-fi"], rating=8.8),
        ]

        writer = LLMRecommendationWriterAgent(llm)
        result = writer.write(
            user_message="Sci-fi please",
            constraints=Constraints(genres=["sci-fi"]),
            candidates=movies,
            rejected_titles=["Inception"],
        )

        assert result is not None
        assert result.movie.title == "The Matrix"
