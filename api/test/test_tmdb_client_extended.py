"""Tests for extended TMDB client functionality.

These tests verify the new person search, keyword search, and
discover-with-person capabilities added to support rich search queries.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.integrations.tmdb_client import TMDBClient, TMDBClientError


class TestTMDBClientPersonSearch:
    """Tests for person search functionality."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        return MagicMock()

    @pytest.fixture
    def tmdb_client(self, mock_http_client):
        """Create a TMDB client with mocked HTTP."""
        client = TMDBClient(api_key="test-key")
        client._client = mock_http_client
        return client

    def test_search_person_found(self, tmdb_client, mock_http_client):
        """Should return person ID when found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 12345, "name": "Angelina Jolie"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        result = tmdb_client.search_person("Angelina Jolie")

        assert result == 12345

    def test_search_person_not_found(self, tmdb_client, mock_http_client):
        """Should return None when person not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        result = tmdb_client.search_person("Unknown Person XYZ")

        assert result is None

    def test_search_persons_multiple(self, tmdb_client, mock_http_client):
        """Should return multiple person IDs."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_response.json.return_value = {
                    "results": [{"id": 111, "name": "Actor One"}]
                }
            else:
                mock_response.json.return_value = {
                    "results": [{"id": 222, "name": "Actor Two"}]
                }
            return mock_response
        
        mock_http_client.get.side_effect = side_effect

        result = tmdb_client.search_persons(["Actor One", "Actor Two"])

        assert result == [111, 222]

    def test_search_person_api_error(self, tmdb_client, mock_http_client):
        """Should return None on API error."""
        import httpx
        mock_http_client.get.side_effect = httpx.RequestError("Connection failed")

        result = tmdb_client.search_person("Some Actor")

        assert result is None


class TestTMDBClientKeywordSearch:
    """Tests for keyword search functionality."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        return MagicMock()

    @pytest.fixture
    def tmdb_client(self, mock_http_client):
        """Create a TMDB client with mocked HTTP."""
        client = TMDBClient(api_key="test-key")
        client._client = mock_http_client
        return client

    def test_search_keyword_found(self, tmdb_client, mock_http_client):
        """Should return keyword ID when found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 9999, "name": "heist"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        result = tmdb_client.search_keyword("heist")

        assert result == 9999

    def test_search_keyword_not_found(self, tmdb_client, mock_http_client):
        """Should return None when keyword not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        result = tmdb_client.search_keyword("nonexistent-keyword-xyz")

        assert result is None

    def test_search_keywords_multiple(self, tmdb_client, mock_http_client):
        """Should return multiple keyword IDs."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_response.json.return_value = {
                    "results": [{"id": 111, "name": "heist"}]
                }
            else:
                mock_response.json.return_value = {
                    "results": [{"id": 222, "name": "robbery"}]
                }
            return mock_response
        
        mock_http_client.get.side_effect = side_effect

        result = tmdb_client.search_keywords(["heist", "robbery"])

        assert result == [111, 222]


class TestTMDBClientDiscoverWithPerson:
    """Tests for discover with person filtering."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        return MagicMock()

    @pytest.fixture
    def tmdb_client(self, mock_http_client):
        """Create a TMDB client with mocked HTTP."""
        client = TMDBClient(api_key="test-key")
        client._client = mock_http_client
        return client

    def test_discover_with_cast_ids(self, tmdb_client, mock_http_client):
        """Should include with_cast parameter in discover."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        tmdb_client.discover_movies(with_cast=[12345, 67890])

        call_args = mock_http_client.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("with_cast") == "12345,67890"

    def test_discover_with_crew_ids(self, tmdb_client, mock_http_client):
        """Should include with_crew parameter in discover."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        tmdb_client.discover_movies(with_crew=[11111])

        call_args = mock_http_client.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("with_crew") == "11111"

    def test_discover_with_year_range(self, tmdb_client, mock_http_client):
        """Should include year range parameters in discover."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        tmdb_client.discover_movies(year_start=1990, year_end=1999)

        call_args = mock_http_client.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("primary_release_date.gte") == "1990-01-01"
        assert params.get("primary_release_date.lte") == "1999-12-31"

    def test_discover_with_keywords(self, tmdb_client, mock_http_client):
        """Should include with_keywords parameter in discover."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        tmdb_client.discover_movies(with_keywords=[100, 200, 300])

        call_args = mock_http_client.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("with_keywords") == "100|200|300"

    def test_discover_with_language(self, tmdb_client, mock_http_client):
        """Should include with_original_language parameter in discover."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        tmdb_client.discover_movies(with_original_language="ko")

        call_args = mock_http_client.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("with_original_language") == "ko"

    def test_discover_year_takes_precedence_over_range(self, tmdb_client, mock_http_client):
        """Exact year should take precedence over year range."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        tmdb_client.discover_movies(year=2020, year_start=2010, year_end=2025)

        call_args = mock_http_client.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("primary_release_year") == 2020
        assert "primary_release_date.gte" not in params
        assert "primary_release_date.lte" not in params


class TestTMDBClientGetPersonMovies:
    """Tests for get_person_movies functionality."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        return MagicMock()

    @pytest.fixture
    def tmdb_client(self, mock_http_client):
        """Create a TMDB client with mocked HTTP."""
        client = TMDBClient(api_key="test-key")
        client._client = mock_http_client
        return client

    def test_get_person_movies_as_cast(self, tmdb_client, mock_http_client):
        """Should return movies where person is in cast."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "cast": [
                {
                    "id": 1,
                    "title": "Movie One",
                    "popularity": 100,
                    "release_date": "2020-01-01",
                    "genre_ids": [28],
                },
                {
                    "id": 2,
                    "title": "Movie Two",
                    "popularity": 50,
                    "release_date": "2019-01-01",
                    "genre_ids": [18],
                },
            ],
            "crew": [],
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        results = tmdb_client.get_person_movies(12345, as_cast=True)

        assert len(results) == 2
        assert results[0].title == "Movie One"

    def test_get_person_movies_as_crew(self, tmdb_client, mock_http_client):
        """Should return movies where person is in crew (director)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "cast": [],
            "crew": [
                {
                    "id": 10,
                    "title": "Directed Movie",
                    "popularity": 200,
                    "release_date": "2021-05-01",
                    "genre_ids": [878],
                    "job": "Director",
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        results = tmdb_client.get_person_movies(54321, as_cast=False)

        assert len(results) == 1
        assert results[0].title == "Directed Movie"

    def test_get_person_movies_sorted_by_popularity(self, tmdb_client, mock_http_client):
        """Should return movies sorted by popularity descending."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "cast": [
                {"id": 1, "title": "Less Popular", "popularity": 10, "release_date": "2020-01-01", "genre_ids": []},
                {"id": 2, "title": "Most Popular", "popularity": 100, "release_date": "2020-01-01", "genre_ids": []},
                {"id": 3, "title": "Medium Popular", "popularity": 50, "release_date": "2020-01-01", "genre_ids": []},
            ],
            "crew": [],
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        results = tmdb_client.get_person_movies(12345)

        assert results[0].title == "Most Popular"
        assert results[1].title == "Medium Popular"
        assert results[2].title == "Less Popular"
