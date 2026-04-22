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

    def search_person(self, name: str) -> int | None:
        """Search for a person (actor/director) by name and return their TMDB ID.

        Args:
            name: Person's name to search for.

        Returns:
            TMDB person ID if found, None otherwise.
        """
        params = {
            "query": name,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }

        try:
            data = self._get("/search/person", params)
            results = data.get("results", [])
            if results:
                person = results[0]
                person_id = person.get("id")
                logger.debug(f"Resolved '{name}' to person ID: {person_id}")
                return person_id
            logger.debug(f"No person found for: {name}")
            return None
        except TMDBClientError as e:
            logger.warning(f"Person search failed for '{name}': {e}")
            return None

    def search_persons(self, names: list[str]) -> list[int]:
        """Search for multiple persons and return their TMDB IDs.

        Args:
            names: List of person names to search for.

        Returns:
            List of TMDB person IDs for found persons.
        """
        ids = []
        for name in names:
            person_id = self.search_person(name)
            if person_id:
                ids.append(person_id)
        return ids

    def discover_movies(
        self,
        genres: list[str] | None = None,
        max_runtime: int | None = None,
        min_runtime: int | None = None,
        min_rating: float | None = None,
        year: int | None = None,
        year_start: int | None = None,
        year_end: int | None = None,
        with_cast: list[int] | None = None,
        with_crew: list[int] | None = None,
        with_keywords: list[int] | None = None,
        with_original_language: str | None = None,
        limit: int = 20,
    ) -> list[MovieResult]:
        """Discover movies using TMDB's discover endpoint.

        Args:
            genres: List of genre names to filter by.
            max_runtime: Maximum runtime in minutes.
            min_runtime: Minimum runtime in minutes.
            min_rating: Minimum vote average (0-10).
            year: Exact release year filter.
            year_start: Start of release year range (inclusive).
            year_end: End of release year range (inclusive).
            with_cast: List of person IDs for cast filtering.
            with_crew: List of person IDs for crew filtering (directors).
            with_keywords: List of keyword IDs for thematic filtering.
            with_original_language: ISO 639-1 language code.
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
        elif year_start or year_end:
            if year_start:
                params["primary_release_date.gte"] = f"{year_start}-01-01"
            if year_end:
                params["primary_release_date.lte"] = f"{year_end}-12-31"

        if with_cast:
            params["with_cast"] = ",".join(str(pid) for pid in with_cast)

        if with_crew:
            params["with_crew"] = ",".join(str(pid) for pid in with_crew)

        if with_keywords:
            params["with_keywords"] = "|".join(str(kid) for kid in with_keywords)

        if with_original_language:
            params["with_original_language"] = with_original_language

        logger.debug(f"TMDB discover params: {params}")
        data = self._get("/discover/movie", params)

        results = []
        for item in data.get("results", [])[:limit]:
            movie = self._normalize_movie(item)
            if movie:
                results.append(movie)

        return results

    def search_keyword(self, keyword: str) -> int | None:
        """Search for a keyword and return its TMDB ID.

        Args:
            keyword: Keyword to search for.

        Returns:
            TMDB keyword ID if found, None otherwise.
        """
        params = {"query": keyword, "page": 1}

        try:
            data = self._get("/search/keyword", params)
            results = data.get("results", [])
            if results:
                keyword_id = results[0].get("id")
                logger.debug(f"Resolved keyword '{keyword}' to ID: {keyword_id}")
                return keyword_id
            return None
        except TMDBClientError as e:
            logger.warning(f"Keyword search failed for '{keyword}': {e}")
            return None

    def search_keywords(self, keywords: list[str]) -> list[int]:
        """Search for multiple keywords and return their TMDB IDs.

        Args:
            keywords: List of keywords to search for.

        Returns:
            List of TMDB keyword IDs for found keywords.
        """
        ids = []
        for kw in keywords:
            keyword_id = self.search_keyword(kw)
            if keyword_id:
                ids.append(keyword_id)
        return ids

    def get_person_movies(
        self,
        person_id: int,
        as_cast: bool = True,
        limit: int = 20,
    ) -> list[MovieResult]:
        """Get movies featuring a specific person.

        Args:
            person_id: TMDB person ID.
            as_cast: If True, get movies where person is in cast; otherwise crew.
            limit: Maximum number of results to return.

        Returns:
            List of MovieResult objects.
        """
        try:
            data = self._get(f"/person/{person_id}/movie_credits")
            key = "cast" if as_cast else "crew"
            credits = data.get(key, [])

            credits_sorted = sorted(
                credits,
                key=lambda x: x.get("popularity", 0),
                reverse=True,
            )

            results = []
            for item in credits_sorted[:limit]:
                movie = self._normalize_movie(item)
                if movie:
                    results.append(movie)

            return results
        except TMDBClientError as e:
            logger.warning(f"Person movies lookup failed: {e}")
            return []

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
