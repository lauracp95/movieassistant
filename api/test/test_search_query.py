"""Tests for the enhanced movie search functionality.

These tests verify that:
1. MovieSearchQuery schema captures rich search signals
2. StubMovieFinderAgent filters by actors, directors, year, etc.
3. TMDBMovieFinderAgent uses multi-step search strategy
4. InputOrchestratorAgent extracts search queries correctly
"""

import pytest
from unittest.mock import MagicMock, patch

from app.schemas.orchestrator import Constraints, MovieSearchQuery
from app.schemas.domain import MovieResult
from app.llm.movie_finder_agent import (
    StubMovieFinderAgent,
    TMDBMovieFinderAgent,
    LANGUAGE_NAME_TO_CODE,
)


class TestMovieSearchQuerySchema:
    """Tests for the MovieSearchQuery schema."""

    def test_empty_query(self):
        """Empty search query should report is_empty=True."""
        query = MovieSearchQuery()
        assert query.is_empty()
        assert not query.has_person_criteria()
        assert not query.has_year_criteria()
        assert not query.has_keyword_criteria()

    def test_actor_criteria(self):
        """Query with actors should have person criteria."""
        query = MovieSearchQuery(actors=["Tom Hanks"])
        assert not query.is_empty()
        assert query.has_person_criteria()

    def test_director_criteria(self):
        """Query with directors should have person criteria."""
        query = MovieSearchQuery(directors=["Christopher Nolan"])
        assert not query.is_empty()
        assert query.has_person_criteria()

    def test_year_criteria(self):
        """Query with year should have year criteria."""
        query = MovieSearchQuery(year=2020)
        assert not query.is_empty()
        assert query.has_year_criteria()

    def test_year_range_criteria(self):
        """Query with year range should have year criteria."""
        query = MovieSearchQuery(year_start=1990, year_end=1999)
        assert not query.is_empty()
        assert query.has_year_criteria()

    def test_keyword_criteria(self):
        """Query with keywords should have keyword criteria."""
        query = MovieSearchQuery(keywords=["heist", "action"])
        assert not query.is_empty()
        assert query.has_keyword_criteria()

    def test_complex_query(self):
        """Complex query with multiple criteria."""
        query = MovieSearchQuery(
            actors=["Angelina Jolie"],
            directors=["Christopher Nolan"],
            year_start=2000,
            year_end=2020,
            keywords=["thriller"],
            mood="dark",
        )
        assert not query.is_empty()
        assert query.has_person_criteria()
        assert query.has_year_criteria()
        assert query.has_keyword_criteria()


class TestStubMovieFinderWithSearchQuery:
    """Tests for StubMovieFinderAgent with rich search queries."""

    def test_find_by_actor_name(self):
        """Should find movies by actor name."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(actors=["Angelina Jolie"])

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert len(results) >= 1
        assert any("Angelina Jolie" in (m.cast or []) for m in results)

    def test_find_by_director_name(self):
        """Should find movies by director name."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(directors=["Christopher Nolan"])

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert len(results) >= 1
        assert all(
            m.director and "nolan" in m.director.lower()
            for m in results
        )

    def test_find_by_exact_year(self):
        """Should find movies by exact year."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(year=1999)

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert len(results) >= 1
        assert all(m.year == 1999 for m in results)

    def test_find_by_year_range_90s(self):
        """Should find movies from the 90s."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(year_start=1990, year_end=1999)

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert len(results) >= 1
        assert all(1990 <= m.year <= 1999 for m in results if m.year)

    def test_combined_actor_and_genre(self):
        """Should find movies matching both actor and genre."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(actors=["Angelina Jolie"])
        constraints = Constraints(genres=["drama"])

        results = finder.find_movies(
            constraints=constraints,
            search_query=query,
        )

        assert len(results) >= 1
        for movie in results:
            assert "drama" in [g.lower() for g in movie.genres]
            assert "Angelina Jolie" in (movie.cast or [])

    def test_combined_director_and_runtime(self):
        """Should find movies matching director and runtime constraint."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(directors=["Christopher Nolan"])
        constraints = Constraints(max_runtime_minutes=150)

        results = finder.find_movies(
            constraints=constraints,
            search_query=query,
        )

        for movie in results:
            assert movie.director and "nolan" in movie.director.lower()
            if movie.runtime_minutes:
                assert movie.runtime_minutes <= 150

    def test_no_results_for_nonexistent_actor(self):
        """Should return empty list for unknown actor."""
        finder = StubMovieFinderAgent()
        query = MovieSearchQuery(actors=["Unknown Actor Name XYZ"])

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert results == []

    def test_backward_compatibility_without_search_query(self):
        """Should work without search_query for backward compatibility."""
        finder = StubMovieFinderAgent()
        constraints = Constraints(genres=["sci-fi"])

        results = finder.find_movies(constraints=constraints)

        assert len(results) >= 1
        assert all("sci-fi" in [g.lower() for g in m.genres] for m in results)


class TestTMDBMovieFinderWithSearchQuery:
    """Tests for TMDBMovieFinderAgent with rich search queries."""

    @pytest.fixture
    def mock_tmdb_client(self):
        """Create a mock TMDB client."""
        return MagicMock()

    def test_person_search_resolves_actor_ids(self, mock_tmdb_client):
        """Should resolve actor names to IDs and use in discover."""
        mock_tmdb_client.search_persons.return_value = [12345]
        mock_tmdb_client.discover_movies.return_value = [
            MovieResult(
                id="tmdb-1",
                title="Test Movie",
                source="tmdb",
            )
        ]
        mock_tmdb_client.search_keywords.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(actors=["Angelina Jolie"])

        results = finder.find_movies(
            constraints=Constraints(genres=["drama"]),
            search_query=query,
        )

        mock_tmdb_client.search_persons.assert_called_once_with(["Angelina Jolie"])
        assert any(
            call.kwargs.get("with_cast") == [12345]
            for call in mock_tmdb_client.discover_movies.call_args_list
        )

    def test_person_search_resolves_director_ids(self, mock_tmdb_client):
        """Should resolve director names to IDs and use in discover."""
        mock_tmdb_client.search_persons.return_value = [54321]
        mock_tmdb_client.discover_movies.return_value = [
            MovieResult(
                id="tmdb-2",
                title="Nolan Movie",
                source="tmdb",
            )
        ]
        mock_tmdb_client.search_keywords.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(directors=["Christopher Nolan"])

        finder.find_movies(
            constraints=Constraints(genres=["sci-fi"]),
            search_query=query,
        )

        mock_tmdb_client.search_persons.assert_called_once_with(["Christopher Nolan"])
        assert any(
            call.kwargs.get("with_crew") == [54321]
            for call in mock_tmdb_client.discover_movies.call_args_list
        )

    def test_year_range_passed_to_discover(self, mock_tmdb_client):
        """Should pass year range to discover endpoint."""
        mock_tmdb_client.discover_movies.return_value = []
        mock_tmdb_client.search_keywords.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(year_start=1990, year_end=1999)

        finder.find_movies(
            constraints=Constraints(genres=["thriller"]),
            search_query=query,
        )

        assert any(
            call.kwargs.get("year_start") == 1990 and call.kwargs.get("year_end") == 1999
            for call in mock_tmdb_client.discover_movies.call_args_list
        )

    def test_keywords_resolved_and_passed(self, mock_tmdb_client):
        """Should resolve keywords to IDs and pass to discover."""
        mock_tmdb_client.discover_movies.return_value = []
        mock_tmdb_client.search_keywords.return_value = [11111, 22222]

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(keywords=["heist", "robbery"])

        finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        mock_tmdb_client.search_keywords.assert_called_once_with(["heist", "robbery"])
        assert any(
            call.kwargs.get("with_keywords") == [11111, 22222]
            for call in mock_tmdb_client.discover_movies.call_args_list
        )

    def test_text_query_uses_search_movies(self, mock_tmdb_client):
        """Should use search_movies for text_query."""
        mock_tmdb_client.discover_movies.return_value = []
        mock_tmdb_client.search_keywords.return_value = []
        mock_tmdb_client.search_movies.return_value = [
            MovieResult(id="tmdb-3", title="Star Wars", source="tmdb")
        ]

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(text_query="Star Wars")

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        mock_tmdb_client.search_movies.assert_called_once_with("Star Wars", limit=10)
        assert len(results) == 1
        assert results[0].title == "Star Wars"

    def test_language_resolution(self, mock_tmdb_client):
        """Should resolve language names to ISO codes."""
        mock_tmdb_client.discover_movies.return_value = []
        mock_tmdb_client.search_keywords.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(language="Korean")

        finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert any(
            call.kwargs.get("with_original_language") == "ko"
            for call in mock_tmdb_client.discover_movies.call_args_list
        )

    def test_deduplicates_results(self, mock_tmdb_client):
        """Should deduplicate results from multiple search strategies."""
        same_movie = MovieResult(id="tmdb-1", title="Same Movie", source="tmdb")
        mock_tmdb_client.search_persons.return_value = [12345]
        mock_tmdb_client.discover_movies.return_value = [same_movie]
        mock_tmdb_client.search_keywords.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)
        query = MovieSearchQuery(actors=["Actor Name"])

        results = finder.find_movies(
            constraints=Constraints(),
            search_query=query,
        )

        assert len(results) == 1

    def test_excludes_rejected_titles(self, mock_tmdb_client):
        """Should exclude rejected titles from results."""
        mock_tmdb_client.discover_movies.return_value = [
            MovieResult(id="tmdb-1", title="Good Movie", source="tmdb"),
            MovieResult(id="tmdb-2", title="Rejected Movie", source="tmdb"),
        ]
        mock_tmdb_client.search_keywords.return_value = []

        finder = TMDBMovieFinderAgent(mock_tmdb_client)

        results = finder.find_movies(
            constraints=Constraints(),
            excluded_titles=["Rejected Movie"],
        )

        assert len(results) == 1
        assert results[0].title == "Good Movie"

    def test_handles_api_errors_gracefully(self, mock_tmdb_client):
        """Should return empty list on API errors."""
        mock_tmdb_client.discover_movies.side_effect = Exception("API Error")

        finder = TMDBMovieFinderAgent(mock_tmdb_client)

        results = finder.find_movies(constraints=Constraints())

        assert results == []


class TestLanguageCodeMapping:
    """Tests for language name to code mapping."""

    def test_common_languages(self):
        """Common language names should map correctly."""
        assert LANGUAGE_NAME_TO_CODE["korean"] == "ko"
        assert LANGUAGE_NAME_TO_CODE["french"] == "fr"
        assert LANGUAGE_NAME_TO_CODE["japanese"] == "ja"
        assert LANGUAGE_NAME_TO_CODE["spanish"] == "es"
        assert LANGUAGE_NAME_TO_CODE["chinese"] == "zh"

    def test_case_insensitive_lookup(self):
        """Should use lowercase keys for lookup."""
        finder = TMDBMovieFinderAgent(MagicMock())
        assert finder._resolve_language("Korean") == "ko"
        assert finder._resolve_language("FRENCH") == "fr"

    def test_iso_code_passthrough(self):
        """Two-letter codes should pass through."""
        finder = TMDBMovieFinderAgent(MagicMock())
        assert finder._resolve_language("ko") == "ko"
        assert finder._resolve_language("en") == "en"

    def test_unknown_language_returns_none(self):
        """Unknown languages should return None."""
        finder = TMDBMovieFinderAgent(MagicMock())
        assert finder._resolve_language("klingon") is None
        assert finder._resolve_language(None) is None


class TestSearchQueryExtractionScenarios:
    """Integration-style tests for expected search extraction behavior.
    
    These tests document the expected extraction for various user requests.
    They verify the schema can represent these queries correctly.
    """

    def test_drama_with_angelina_jolie_under_2_hours(self):
        """Scenario: 'drama movie with Angelina Jolie under 2 hours'"""
        constraints = Constraints(genres=["drama"], max_runtime_minutes=120)
        query = MovieSearchQuery(actors=["Angelina Jolie"])

        assert constraints.genres == ["drama"]
        assert constraints.max_runtime_minutes == 120
        assert query.actors == ["Angelina Jolie"]
        assert query.has_person_criteria()

    def test_scifi_by_christopher_nolan(self):
        """Scenario: 'sci-fi movie by Christopher Nolan'"""
        constraints = Constraints(genres=["sci-fi"])
        query = MovieSearchQuery(directors=["Christopher Nolan"])

        assert constraints.genres == ["sci-fi"]
        assert query.directors == ["Christopher Nolan"]
        assert query.has_person_criteria()

    def test_90s_thriller(self):
        """Scenario: '90s thriller'"""
        constraints = Constraints(genres=["thriller"])
        query = MovieSearchQuery(year_start=1990, year_end=1999)

        assert constraints.genres == ["thriller"]
        assert query.year_start == 1990
        assert query.year_end == 1999
        assert query.has_year_criteria()

    def test_dark_slow_drama_under_2_hours(self):
        """Scenario: 'dark, slow drama with strong female lead, under 2 hours'"""
        constraints = Constraints(genres=["drama"], max_runtime_minutes=120)
        query = MovieSearchQuery(
            mood="dark",
            keywords=["slow", "strong female lead"],
        )

        assert constraints.genres == ["drama"]
        assert constraints.max_runtime_minutes == 120
        assert query.mood == "dark"
        assert "slow" in query.keywords
        assert "strong female lead" in query.keywords

    def test_korean_horror(self):
        """Scenario: 'Korean horror movie'"""
        constraints = Constraints(genres=["horror"])
        query = MovieSearchQuery(language="ko")

        assert constraints.genres == ["horror"]
        assert query.language == "ko"

    def test_heist_movie_like_oceans_eleven(self):
        """Scenario: 'heist movie like Ocean's Eleven'"""
        constraints = Constraints()
        query = MovieSearchQuery(
            keywords=["heist"],
            text_query="Ocean's Eleven",
        )

        assert "heist" in query.keywords
        assert query.text_query == "Ocean's Eleven"

    def test_recent_action_with_tom_cruise(self):
        """Scenario: 'recent action movie with Tom Cruise'"""
        constraints = Constraints(genres=["action"])
        query = MovieSearchQuery(
            actors=["Tom Cruise"],
            year_start=2020,
        )

        assert constraints.genres == ["action"]
        assert "Tom Cruise" in query.actors
        assert query.year_start == 2020
        assert query.has_year_criteria()
        assert query.has_person_criteria()
