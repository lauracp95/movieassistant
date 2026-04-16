"""TMDB API client for movie data retrieval.

This module provides a client for the TMDB (The Movie Database) API,
handling authentication, request formatting, and response normalization
to the internal MovieResult model.
"""

import logging
from typing import Any

import httpx

from app.schemas.domain import MovieResult

logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

GENRE_NAME_TO_ID: dict[str, int] = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "sci-fi": 878,
    "tv movie": 10770,
    "thriller": 53,
    "war": 10752,
    "western": 37,
}

GENRE_ID_TO_NAME: dict[int, str] = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


class TMDBClientError(Exception):
    """Exception raised for TMDB API errors."""

    pass


class TMDBClient:
    """Client for TMDB API interactions.

    Handles authentication and provides methods for discovering
    and searching movies with constraints.
    """

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        """Initialize the TMDB client.

        Args:
            api_key: TMDB API key (v3 auth).
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "TMDBClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Make a GET request to the TMDB API.

        Args:
            endpoint: API endpoint path (without base URL).
            params: Query parameters.

        Returns:
            JSON response as a dictionary.

        Raises:
            TMDBClientError: If the request fails.
        """
        url = f"{TMDB_BASE_URL}{endpoint}"
        request_params = {"api_key": self._api_key}
        if params:
            request_params.update(params)

        try:
            response = self._client.get(url, params=request_params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"TMDB API error: {e.response.status_code} - {e.response.text}")
            raise TMDBClientError(f"TMDB API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"TMDB request failed: {e}")
            raise TMDBClientError(f"TMDB request failed: {e}") from e

    def discover_movies(
        self,
        genres: list[str] | None = None,
        max_runtime: int | None = None,
        min_runtime: int | None = None,
        min_rating: float | None = None,
        year: int | None = None,
        limit: int = 20,
    ) -> list[MovieResult]:
        """Discover movies using TMDB's discover endpoint.

        Args:
            genres: List of genre names to filter by.
            max_runtime: Maximum runtime in minutes.
            min_runtime: Minimum runtime in minutes.
            min_rating: Minimum vote average (0-10).
            year: Release year filter.
            limit: Maximum number of results to return.

        Returns:
            List of MovieResult objects matching the criteria.
        """
        params: dict[str, Any] = {
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "language": "en-US",
            "page": 1,
        }

        if genres:
            genre_ids = self._resolve_genre_ids(genres)
            if genre_ids:
                params["with_genres"] = "|".join(str(g) for g in genre_ids)

        if max_runtime:
            params["with_runtime.lte"] = max_runtime

        if min_runtime:
            params["with_runtime.gte"] = min_runtime

        if min_rating:
            params["vote_average.gte"] = min_rating

        if year:
            params["primary_release_year"] = year

        logger.debug(f"TMDB discover params: {params}")
        data = self._get("/discover/movie", params)

        results = []
        for item in data.get("results", [])[:limit]:
            movie = self._normalize_movie(item)
            if movie:
                results.append(movie)

        return results

    def search_movies(self, query: str, limit: int = 20) -> list[MovieResult]:
        """Search for movies by title.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of MovieResult objects matching the search.
        """
        params = {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }

        data = self._get("/search/movie", params)

        results = []
        for item in data.get("results", [])[:limit]:
            movie = self._normalize_movie(item)
            if movie:
                results.append(movie)

        return results

    def get_movie_details(self, movie_id: int) -> MovieResult | None:
        """Get detailed information about a specific movie.

        Args:
            movie_id: TMDB movie ID.

        Returns:
            MovieResult with full details, or None if not found.
        """
        try:
            data = self._get(f"/movie/{movie_id}")
            return self._normalize_movie_details(data)
        except TMDBClientError:
            return None

    def _resolve_genre_ids(self, genre_names: list[str]) -> list[int]:
        """Convert genre names to TMDB genre IDs."""
        ids = []
        for name in genre_names:
            genre_id = GENRE_NAME_TO_ID.get(name.lower())
            if genre_id:
                ids.append(genre_id)
            else:
                logger.warning(f"Unknown genre: {name}")
        return ids

    def _normalize_movie(self, data: dict) -> MovieResult | None:
        """Normalize TMDB movie response to MovieResult.

        Handles missing or malformed data gracefully.

        Args:
            data: Raw movie data from TMDB API.

        Returns:
            MovieResult or None if data is invalid.
        """
        try:
            movie_id = data.get("id")
            title = data.get("title")

            if not movie_id or not title:
                return None

            release_date = data.get("release_date", "")
            year = None
            if release_date and len(release_date) >= 4:
                try:
                    year = int(release_date[:4])
                except ValueError:
                    pass

            genre_ids = data.get("genre_ids", [])
            genres = [GENRE_ID_TO_NAME.get(gid, "Unknown") for gid in genre_ids]
            genres = [g for g in genres if g != "Unknown"]

            poster_path = data.get("poster_path")
            poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None

            return MovieResult(
                id=f"tmdb-{movie_id}",
                title=title,
                year=year,
                genres=genres,
                runtime_minutes=None,
                overview=data.get("overview"),
                rating=data.get("vote_average"),
                poster_url=poster_url,
                source="tmdb",
            )
        except Exception as e:
            logger.warning(f"Failed to normalize movie data: {e}")
            return None

    def _normalize_movie_details(self, data: dict) -> MovieResult | None:
        """Normalize detailed movie response (includes runtime).

        Args:
            data: Raw movie details from TMDB API.

        Returns:
            MovieResult with runtime, or None if data is invalid.
        """
        try:
            movie_id = data.get("id")
            title = data.get("title")

            if not movie_id or not title:
                return None

            release_date = data.get("release_date", "")
            year = None
            if release_date and len(release_date) >= 4:
                try:
                    year = int(release_date[:4])
                except ValueError:
                    pass

            genres_data = data.get("genres", [])
            genres = [g.get("name") for g in genres_data if g.get("name")]

            poster_path = data.get("poster_path")
            poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None

            return MovieResult(
                id=f"tmdb-{movie_id}",
                title=title,
                year=year,
                genres=genres,
                runtime_minutes=data.get("runtime"),
                overview=data.get("overview"),
                rating=data.get("vote_average"),
                poster_url=poster_url,
                source="tmdb",
            )
        except Exception as e:
            logger.warning(f"Failed to normalize movie details: {e}")
            return None
