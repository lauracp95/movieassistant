"""Unit tests for MovieFinderAgent implementations."""

from unittest.mock import MagicMock, patch

import pytest

from app.integrations.tmdb_client import TMDBClient, TMDBClientError
from app.llm.movie_finder_agent import (
    MovieFinderAgent,
    StubMovieFinderAgent,
    TMDBMovieFinderAgent,
)
from app.schemas.domain import MovieResult
from app.schemas.orchestrator import Constraints


class TestStubMovieFinderAgent:
    def test_stub_finder_returns_movie_results(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints())

        assert len(results) > 0
        assert all(isinstance(m, MovieResult) for m in results)

    def test_stub_finder_filters_by_genre(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints(genres=["Horror"]))

        assert len(results) > 0
        for movie in results:
            movie_genres = [g.lower() for g in movie.genres]
            assert "horror" in movie_genres

    def test_stub_finder_filters_by_multiple_genres(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints(genres=["Action", "Sci-Fi"]))

        assert len(results) > 0
        for movie in results:
            movie_genres = {g.lower() for g in movie.genres}
            assert movie_genres & {"action", "sci-fi"}

    def test_stub_finder_filters_by_max_runtime(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints(max_runtime_minutes=100))

        assert len(results) > 0
        for movie in results:
            assert movie.runtime_minutes is None or movie.runtime_minutes <= 100

    def test_stub_finder_filters_by_min_runtime(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints(min_runtime_minutes=140))

        assert len(results) > 0
        for movie in results:
            assert movie.runtime_minutes is None or movie.runtime_minutes >= 140

    def test_stub_finder_respects_limit(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints(), limit=3)

        assert len(results) <= 3

    def test_stub_finder_excludes_titles(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(
            Constraints(),
            excluded_titles=["The Matrix", "Inception"],
        )

        titles = [m.title.lower() for m in results]
        assert "the matrix" not in titles
        assert "inception" not in titles

    def test_stub_finder_returns_empty_for_no_matches(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(
            Constraints(genres=["NonExistentGenre123"])
        )

        assert results == []

    def test_stub_finder_with_custom_movies(self):
        custom_movies = [
            MovieResult(
                id="custom-1",
                title="Custom Movie",
                year=2024,
                genres=["Custom"],
                source="test",
            ),
        ]
        finder = StubMovieFinderAgent(custom_movies=custom_movies)
        results = finder.find_movies(Constraints())

        assert len(results) == 1
        assert results[0].title == "Custom Movie"

    def test_stub_finder_all_movies_have_source_stub(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(Constraints())

        for movie in results:
            assert movie.source == "stub"

    def test_stub_finder_combined_constraints(self):
        finder = StubMovieFinderAgent()
        results = finder.find_movies(
            Constraints(
                genres=["Sci-Fi"],
                max_runtime_minutes=150,
            )
        )

        assert len(results) > 0
        for movie in results:
            movie_genres = [g.lower() for g in movie.genres]
            assert "sci-fi" in movie_genres
            assert movie.runtime_minutes is None or movie.runtime_minutes <= 150


class TestTMDBMovieFinderAgent:
    @pytest.fixture
    def mock_tmdb_client(self):
        return MagicMock(spec=TMDBClient)

    def test_tmdb_finder_calls_discover_movies(self, mock_tmdb_client):
        mock_tmdb_client.discover_movies.return_value = [
            MovieResult(
                id="tmdb-123",
                title="Test Movie",
                year=2024,
                genres=["Action"],
                source="tmdb",
            ),
        ]

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        results = finder.find_movies(Constraints(genres=["Action"]))

        mock_tmdb_client.discover_movies.assert_called_once()
        assert len(results) == 1
        assert results[0].title == "Test Movie"

    def test_tmdb_finder_passes_constraints(self, mock_tmdb_client):
        mock_tmdb_client.discover_movies.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        finder.find_movies(
            Constraints(
                genres=["Horror", "Thriller"],
                max_runtime_minutes=120,
                min_runtime_minutes=90,
            ),
            limit=5,
        )

        mock_tmdb_client.discover_movies.assert_called_once_with(
            genres=["Horror", "Thriller"],
            max_runtime=120,
            min_runtime=90,
            limit=5,
        )

    def test_tmdb_finder_excludes_titles(self, mock_tmdb_client):
        mock_tmdb_client.discover_movies.return_value = [
            MovieResult(id="tmdb-1", title="Keep Me", genres=[], source="tmdb"),
            MovieResult(id="tmdb-2", title="Exclude Me", genres=[], source="tmdb"),
            MovieResult(id="tmdb-3", title="Keep Too", genres=[], source="tmdb"),
        ]

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        results = finder.find_movies(
            Constraints(),
            excluded_titles=["Exclude Me"],
        )

        titles = [m.title for m in results]
        assert "Keep Me" in titles
        assert "Exclude Me" not in titles
        assert "Keep Too" in titles

    def test_tmdb_finder_respects_limit_after_exclusion(self, mock_tmdb_client):
        mock_tmdb_client.discover_movies.return_value = [
            MovieResult(id="tmdb-1", title="Movie 1", genres=[], source="tmdb"),
            MovieResult(id="tmdb-2", title="Movie 2", genres=[], source="tmdb"),
            MovieResult(id="tmdb-3", title="Movie 3", genres=[], source="tmdb"),
        ]

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        results = finder.find_movies(Constraints(), limit=2)

        assert len(results) == 2

    def test_tmdb_finder_handles_api_error_gracefully(self, mock_tmdb_client):
        mock_tmdb_client.discover_movies.side_effect = TMDBClientError("API Error")

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        results = finder.find_movies(Constraints())

        assert results == []

    def test_tmdb_finder_handles_generic_exception(self, mock_tmdb_client):
        mock_tmdb_client.discover_movies.side_effect = Exception("Unexpected error")

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        results = finder.find_movies(Constraints())

        assert results == []


class TestMovieFinderAgentProtocol:
    def test_stub_finder_is_movie_finder_agent(self):
        finder = StubMovieFinderAgent()
        assert isinstance(finder, MovieFinderAgent)

    def test_tmdb_finder_is_movie_finder_agent(self):
        mock_client = MagicMock(spec=TMDBClient)
        finder = TMDBMovieFinderAgent(mock_client)
        assert isinstance(finder, MovieFinderAgent)
