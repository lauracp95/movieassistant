"""Unit tests for TMDB client."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.integrations.tmdb_client import (
    GENRE_ID_TO_NAME,
    GENRE_NAME_TO_ID,
    TMDBClient,
    TMDBClientError,
)
from app.schemas.domain import MovieResult


class TestTMDBClientNormalization:
    @pytest.fixture
    def client(self):
        return TMDBClient(api_key="test-key")

    def test_normalize_movie_basic(self, client):
        raw_data = {
            "id": 550,
            "title": "Fight Club",
            "release_date": "1999-10-15",
            "genre_ids": [18, 53],
            "overview": "A depressed man suffering from insomnia meets a strange soap salesman.",
            "vote_average": 8.4,
            "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        }

        result = client._normalize_movie(raw_data)

        assert result is not None
        assert result.id == "tmdb-550"
        assert result.title == "Fight Club"
        assert result.year == 1999
        assert "Drama" in result.genres
        assert "Thriller" in result.genres
        assert result.overview.startswith("A depressed man")
        assert result.rating == 8.4
        assert result.poster_url is not None
        assert result.source == "tmdb"

    def test_normalize_movie_missing_optional_fields(self, client):
        raw_data = {
            "id": 123,
            "title": "Minimal Movie",
        }

        result = client._normalize_movie(raw_data)

        assert result is not None
        assert result.id == "tmdb-123"
        assert result.title == "Minimal Movie"
        assert result.year is None
        assert result.genres == []
        assert result.overview is None
        assert result.rating is None
        assert result.poster_url is None

    def test_normalize_movie_invalid_release_date(self, client):
        raw_data = {
            "id": 456,
            "title": "Bad Date Movie",
            "release_date": "invalid-date",
        }

        result = client._normalize_movie(raw_data)

        assert result is not None
        assert result.year is None

    def test_normalize_movie_empty_release_date(self, client):
        raw_data = {
            "id": 789,
            "title": "No Date Movie",
            "release_date": "",
        }

        result = client._normalize_movie(raw_data)

        assert result is not None
        assert result.year is None

    def test_normalize_movie_missing_id_returns_none(self, client):
        raw_data = {"title": "No ID Movie"}

        result = client._normalize_movie(raw_data)

        assert result is None

    def test_normalize_movie_missing_title_returns_none(self, client):
        raw_data = {"id": 999}

        result = client._normalize_movie(raw_data)

        assert result is None

    def test_normalize_movie_unknown_genre_id(self, client):
        raw_data = {
            "id": 111,
            "title": "Unknown Genre Movie",
            "genre_ids": [99999, 18],
        }

        result = client._normalize_movie(raw_data)

        assert result is not None
        assert "Drama" in result.genres
        assert len(result.genres) == 1

    def test_normalize_movie_details_with_runtime(self, client):
        raw_data = {
            "id": 550,
            "title": "Fight Club",
            "release_date": "1999-10-15",
            "genres": [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
            "runtime": 139,
            "overview": "A depressed man...",
            "vote_average": 8.4,
            "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        }

        result = client._normalize_movie_details(raw_data)

        assert result is not None
        assert result.runtime_minutes == 139
        assert "Drama" in result.genres
        assert "Thriller" in result.genres


class TestTMDBClientGenreMapping:
    def test_genre_name_to_id_mapping(self):
        assert GENRE_NAME_TO_ID["action"] == 28
        assert GENRE_NAME_TO_ID["comedy"] == 35
        assert GENRE_NAME_TO_ID["horror"] == 27
        assert GENRE_NAME_TO_ID["sci-fi"] == 878

    def test_genre_id_to_name_mapping(self):
        assert GENRE_ID_TO_NAME[28] == "Action"
        assert GENRE_ID_TO_NAME[35] == "Comedy"
        assert GENRE_ID_TO_NAME[27] == "Horror"
        assert GENRE_ID_TO_NAME[878] == "Sci-Fi"

    def test_resolve_genre_ids(self):
        client = TMDBClient(api_key="test-key")
        ids = client._resolve_genre_ids(["Action", "Comedy", "Horror"])

        assert 28 in ids
        assert 35 in ids
        assert 27 in ids

    def test_resolve_genre_ids_case_insensitive(self):
        client = TMDBClient(api_key="test-key")
        ids = client._resolve_genre_ids(["ACTION", "comedy", "HoRrOr"])

        assert 28 in ids
        assert 35 in ids
        assert 27 in ids

    def test_resolve_genre_ids_unknown_genre(self):
        client = TMDBClient(api_key="test-key")
        ids = client._resolve_genre_ids(["Action", "UnknownGenre123"])

        assert 28 in ids
        assert len(ids) == 1


class TestTMDBClientAPIRequests:
    @pytest.fixture
    def mock_response(self):
        response = MagicMock()
        response.json.return_value = {
            "results": [
                {
                    "id": 550,
                    "title": "Fight Club",
                    "release_date": "1999-10-15",
                    "genre_ids": [18],
                    "overview": "Test overview",
                    "vote_average": 8.4,
                }
            ]
        }
        response.raise_for_status = MagicMock()
        return response

    @patch.object(httpx.Client, "get")
    def test_discover_movies_success(self, mock_get, mock_response):
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        results = client.discover_movies(genres=["Drama"])

        assert len(results) == 1
        assert results[0].title == "Fight Club"
        mock_get.assert_called_once()

    @patch.object(httpx.Client, "get")
    def test_discover_movies_with_filters(self, mock_get, mock_response):
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        client.discover_movies(
            genres=["Action"],
            max_runtime=120,
            min_runtime=90,
            min_rating=7.0,
            year=2023,
        )

        call_args = mock_get.call_args
        params = call_args[1]["params"]

        assert "with_runtime.lte" in params
        assert params["with_runtime.lte"] == 120
        assert "with_runtime.gte" in params
        assert params["with_runtime.gte"] == 90
        assert "vote_average.gte" in params
        assert params["vote_average.gte"] == 7.0
        assert "primary_release_year" in params
        assert params["primary_release_year"] == 2023

    @patch.object(httpx.Client, "get")
    def test_search_movies_success(self, mock_get, mock_response):
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        results = client.search_movies("Fight Club")

        assert len(results) == 1
        assert results[0].title == "Fight Club"

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["query"] == "Fight Club"

    @patch.object(httpx.Client, "get")
    def test_api_error_raises_tmdb_client_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(status_code=401, text="Unauthorized"),
        )
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")

        with pytest.raises(TMDBClientError):
            client.discover_movies()

    @patch.object(httpx.Client, "get")
    def test_request_error_raises_tmdb_client_error(self, mock_get):
        mock_get.side_effect = httpx.RequestError("Connection failed")

        client = TMDBClient(api_key="test-key")

        with pytest.raises(TMDBClientError):
            client.discover_movies()

    @patch.object(httpx.Client, "get")
    def test_empty_results_returns_empty_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        results = client.discover_movies()

        assert results == []

    @patch.object(httpx.Client, "get")
    def test_respects_limit(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": i + 1, "title": f"Movie {i + 1}"} for i in range(20)
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        results = client.discover_movies(limit=5)

        assert len(results) == 5


class TestTMDBClientMalformedData:
    @patch.object(httpx.Client, "get")
    def test_handles_malformed_results(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Valid Movie"},
                {"title": "Missing ID"},
                {"id": 3},
                {"id": 4, "title": "Another Valid"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        results = client.discover_movies()

        assert len(results) == 2
        titles = [r.title for r in results]
        assert "Valid Movie" in titles
        assert "Another Valid" in titles

    @patch.object(httpx.Client, "get")
    def test_handles_missing_results_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"page": 1}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = TMDBClient(api_key="test-key")
        results = client.discover_movies()

        assert results == []
