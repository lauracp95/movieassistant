"""MovieFinderAgent implementations for candidate movie retrieval.

This module provides the movie finder agents that retrieve candidate movies
from external sources. The finder is responsible for fetching raw movie data
based on user constraints, not for ranking or final selection.

Implementations:
- StubMovieFinderAgent: Returns predictable data for tests
- TMDBMovieFinderAgent: Retrieves movies from TMDB API
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.schemas.domain import MovieResult
from app.schemas.orchestrator import Constraints

if TYPE_CHECKING:
    from app.integrations.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)


class MovieFinderAgent(ABC):
    """Abstract base class for movie finder agents.

    Movie finders retrieve candidate movies from external sources based on
    user constraints. They normalize results to MovieResult objects and
    handle error cases gracefully.
    """

    @abstractmethod
    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
    ) -> list[MovieResult]:
        """Find candidate movies matching the given constraints.

        Args:
            constraints: User constraints (genres, runtime limits, etc.)
            limit: Maximum number of movies to return.
            excluded_titles: Titles to exclude from results (e.g., rejected movies).

        Returns:
            List of MovieResult objects. May be empty if no matches found.
            Results are not ranked; ranking is done by a separate agent.
        """
        pass


class StubMovieFinderAgent(MovieFinderAgent):
    """Stub movie finder for testing and local development.

    Returns predictable movie data without making external API calls.
    Supports optional custom movie data for specific test scenarios.
    """

    STUB_MOVIES: list[MovieResult] = [
        MovieResult(
            id="stub-1",
            title="The Matrix",
            year=1999,
            genres=["Action", "Sci-Fi"],
            runtime_minutes=136,
            overview="A computer hacker learns about the true nature of reality.",
            rating=8.7,
            source="stub",
        ),
        MovieResult(
            id="stub-2",
            title="Inception",
            year=2010,
            genres=["Action", "Sci-Fi", "Thriller"],
            runtime_minutes=148,
            overview="A thief who enters dreams to steal secrets.",
            rating=8.8,
            source="stub",
        ),
        MovieResult(
            id="stub-3",
            title="The Conjuring",
            year=2013,
            genres=["Horror", "Mystery", "Thriller"],
            runtime_minutes=112,
            overview="Paranormal investigators help a family terrorized by a dark presence.",
            rating=7.5,
            source="stub",
        ),
        MovieResult(
            id="stub-4",
            title="Superbad",
            year=2007,
            genres=["Comedy"],
            runtime_minutes=113,
            overview="Two co-dependent high school seniors must separate for college.",
            rating=7.6,
            source="stub",
        ),
        MovieResult(
            id="stub-5",
            title="The Notebook",
            year=2004,
            genres=["Drama", "Romance"],
            runtime_minutes=123,
            overview="A poor yet passionate young man falls in love with a rich young woman.",
            rating=7.8,
            source="stub",
        ),
        MovieResult(
            id="stub-6",
            title="Blade Runner 2049",
            year=2017,
            genres=["Action", "Drama", "Sci-Fi"],
            runtime_minutes=164,
            overview="A young blade runner uncovers a long-buried secret.",
            rating=8.0,
            source="stub",
        ),
        MovieResult(
            id="stub-7",
            title="Get Out",
            year=2017,
            genres=["Horror", "Mystery", "Thriller"],
            runtime_minutes=104,
            overview="A young Black man visits his white girlfriend's family estate.",
            rating=7.7,
            source="stub",
        ),
        MovieResult(
            id="stub-8",
            title="The Grand Budapest Hotel",
            year=2014,
            genres=["Adventure", "Comedy", "Crime"],
            runtime_minutes=99,
            overview="A concierge and his lobby boy are caught up in a murder mystery.",
            rating=8.1,
            source="stub",
        ),
    ]

    def __init__(self, custom_movies: list[MovieResult] | None = None) -> None:
        """Initialize the stub finder.

        Args:
            custom_movies: Optional list of movies to use instead of defaults.
        """
        self._movies = custom_movies if custom_movies is not None else self.STUB_MOVIES

    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
    ) -> list[MovieResult]:
        """Find movies matching constraints from stub data.

        Applies genre and runtime filtering to the stub movie list.

        Args:
            constraints: User constraints to filter by.
            limit: Maximum number of movies to return.
            excluded_titles: Titles to exclude from results.

        Returns:
            List of matching MovieResult objects.
        """
        logger.info(f"StubMovieFinder searching with constraints: {constraints}")
        excluded = set(t.lower() for t in (excluded_titles or []))

        results = []
        for movie in self._movies:
            if movie.title.lower() in excluded:
                continue

            if not self._matches_constraints(movie, constraints):
                continue

            results.append(movie)

            if len(results) >= limit:
                break

        logger.info(f"StubMovieFinder found {len(results)} movies")
        return results

    def _matches_constraints(self, movie: MovieResult, constraints: Constraints) -> bool:
        """Check if a movie matches the given constraints."""
        if constraints.genres:
            constraint_genres = {g.lower() for g in constraints.genres}
            movie_genres = {g.lower() for g in movie.genres}
            if not constraint_genres & movie_genres:
                return False

        if constraints.max_runtime_minutes and movie.runtime_minutes:
            if movie.runtime_minutes > constraints.max_runtime_minutes:
                return False

        if constraints.min_runtime_minutes and movie.runtime_minutes:
            if movie.runtime_minutes < constraints.min_runtime_minutes:
                return False

        return True


class TMDBMovieFinderAgent(MovieFinderAgent):
    """Movie finder that retrieves candidates from TMDB API.

    Uses the TMDB discover and search endpoints to find movies
    matching user constraints.
    """

    def __init__(self, tmdb_client: "TMDBClient") -> None:
        """Initialize with a TMDB client.

        Args:
            tmdb_client: Configured TMDBClient instance.
        """
        self._client = tmdb_client

    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
    ) -> list[MovieResult]:
        """Find movies matching constraints from TMDB.

        Args:
            constraints: User constraints to search by.
            limit: Maximum number of movies to return.
            excluded_titles: Titles to exclude from results.

        Returns:
            List of matching MovieResult objects from TMDB.
        """
        logger.info(f"TMDBMovieFinder searching with constraints: {constraints}")
        excluded = set(t.lower() for t in (excluded_titles or []))

        try:
            raw_movies = self._client.discover_movies(
                genres=constraints.genres,
                max_runtime=constraints.max_runtime_minutes,
                min_runtime=constraints.min_runtime_minutes,
                limit=limit + len(excluded),
            )

            results = []
            for movie in raw_movies:
                if movie.title.lower() in excluded:
                    continue
                results.append(movie)
                if len(results) >= limit:
                    break

            logger.info(f"TMDBMovieFinder found {len(results)} movies")
            return results

        except Exception as e:
            logger.error(f"TMDB search failed: {e}")
            return []
