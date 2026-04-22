"""MovieFinderAgent implementations for candidate movie retrieval.

This module provides the movie finder agents that retrieve candidate movies
from external sources. The finder is responsible for fetching raw movie data
based on user constraints, not for ranking or final selection.

Implementations:
- StubMovieFinderAgent: Returns predictable data for tests
- TMDBMovieFinderAgent: Retrieves movies from TMDB API with rich search
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.schemas.domain import MovieResult
from app.schemas.orchestrator import Constraints, MovieSearchQuery

if TYPE_CHECKING:
    from app.integrations.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)

LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "zh",
    "hindi": "hi",
    "arabic": "ar",
    "russian": "ru",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "finnish": "fi",
    "dutch": "nl",
    "polish": "pl",
    "thai": "th",
    "turkish": "tr",
}


class MovieFinderAgent(ABC):
    """Abstract base class for movie finder agents.

    Movie finders retrieve candidate movies from external sources based on
    user constraints and search queries. They normalize results to MovieResult
    objects and handle error cases gracefully.
    """

    @abstractmethod
    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
        search_query: MovieSearchQuery | None = None,
    ) -> list[MovieResult]:
        """Find candidate movies matching the given constraints and search query.

        Args:
            constraints: Hard constraints (genres, runtime limits) for filtering.
            limit: Maximum number of movies to return.
            excluded_titles: Titles to exclude from results (e.g., rejected movies).
            search_query: Rich search query with actors, directors, year, keywords, etc.

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
            cast=["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
            director="Lana Wachowski",
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
            cast=["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Ellen Page"],
            director="Christopher Nolan",
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
            cast=["Vera Farmiga", "Patrick Wilson"],
            director="James Wan",
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
            cast=["Jonah Hill", "Michael Cera", "Seth Rogen"],
            director="Greg Mottola",
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
            cast=["Ryan Gosling", "Rachel McAdams"],
            director="Nick Cassavetes",
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
            cast=["Ryan Gosling", "Harrison Ford", "Ana de Armas"],
            director="Denis Villeneuve",
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
            cast=["Daniel Kaluuya", "Allison Williams"],
            director="Jordan Peele",
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
            cast=["Ralph Fiennes", "Tony Revolori", "Saoirse Ronan"],
            director="Wes Anderson",
        ),
        MovieResult(
            id="stub-9",
            title="Girl, Interrupted",
            year=1999,
            genres=["Drama"],
            runtime_minutes=127,
            overview="A young woman with depression is admitted to a psychiatric hospital.",
            rating=7.3,
            source="stub",
            cast=["Winona Ryder", "Angelina Jolie"],
            director="James Mangold",
        ),
        MovieResult(
            id="stub-10",
            title="Interstellar",
            year=2014,
            genres=["Adventure", "Drama", "Sci-Fi"],
            runtime_minutes=169,
            overview="A team of explorers travel through a wormhole in space.",
            rating=8.6,
            source="stub",
            cast=["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
            director="Christopher Nolan",
        ),
        MovieResult(
            id="stub-11",
            title="The Silence of the Lambs",
            year=1991,
            genres=["Thriller", "Crime", "Drama"],
            runtime_minutes=118,
            overview="A young FBI cadet must receive the help of an incarcerated cannibal killer.",
            rating=8.6,
            source="stub",
            cast=["Jodie Foster", "Anthony Hopkins"],
            director="Jonathan Demme",
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
        search_query: MovieSearchQuery | None = None,
    ) -> list[MovieResult]:
        """Find movies matching constraints and search query from stub data.

        Applies genre, runtime, and rich search filtering to the stub movie list.

        Args:
            constraints: Hard constraints to filter by (genres, runtime).
            limit: Maximum number of movies to return.
            excluded_titles: Titles to exclude from results.
            search_query: Rich search query with actors, directors, year, etc.

        Returns:
            List of matching MovieResult objects.
        """
        logger.info(f"StubMovieFinder searching with constraints: {constraints}")
        if search_query:
            logger.info(f"StubMovieFinder search_query: {search_query.model_dump_json()}")

        excluded = set(t.lower() for t in (excluded_titles or []))

        results = []
        for movie in self._movies:
            if movie.title.lower() in excluded:
                continue

            if not self._matches_constraints(movie, constraints):
                continue

            if search_query and not self._matches_search_query(movie, search_query):
                continue

            results.append(movie)

            if len(results) >= limit:
                break

        logger.info(f"StubMovieFinder found {len(results)} movies")
        return results

    def _matches_constraints(self, movie: MovieResult, constraints: Constraints) -> bool:
        """Check if a movie matches the given hard constraints."""
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

    def _matches_search_query(
        self, movie: MovieResult, query: MovieSearchQuery
    ) -> bool:
        """Check if a movie matches the rich search query."""
        if query.actors:
            if not movie.cast:
                return False
            movie_cast_lower = [c.lower() for c in movie.cast]
            if not any(
                actor.lower() in " ".join(movie_cast_lower)
                for actor in query.actors
            ):
                return False

        if query.directors:
            if not movie.director:
                return False
            movie_director_lower = movie.director.lower()
            if not any(
                director.lower() in movie_director_lower
                for director in query.directors
            ):
                return False

        if query.year:
            if movie.year != query.year:
                return False

        if query.year_start or query.year_end:
            if not movie.year:
                return False
            if query.year_start and movie.year < query.year_start:
                return False
            if query.year_end and movie.year > query.year_end:
                return False

        return True


class TMDBMovieFinderAgent(MovieFinderAgent):
    """Movie finder that retrieves candidates from TMDB API.

    Uses a multi-step search strategy based on the richness of the search query:
    1. If actors/directors specified → resolve to person IDs → discover with person filter
    2. If keywords specified → resolve to keyword IDs → discover with keywords
    3. If year/period specified → add year filters to discover
    4. Falls back to genre-based discover when no rich criteria available

    The search strategy aims to find the most relevant candidates for the
    user's natural language request, not just generic popular movies.
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
        search_query: MovieSearchQuery | None = None,
    ) -> list[MovieResult]:
        """Find movies matching constraints and search query from TMDB.

        Uses a multi-step search strategy:
        1. Person-based search if actors/directors specified
        2. Keyword-enhanced discover if keywords specified
        3. Year-filtered discover if year/period specified
        4. Basic genre discover as fallback

        Results are merged and deduplicated to maximize relevance.

        Args:
            constraints: Hard constraints (genres, runtime) for filtering.
            limit: Maximum number of movies to return.
            excluded_titles: Titles to exclude from results.
            search_query: Rich search query with actors, directors, year, keywords, etc.

        Returns:
            List of matching MovieResult objects from TMDB.
        """
        logger.info(f"TMDBMovieFinder searching with constraints: {constraints}")
        if search_query:
            logger.info(f"TMDBMovieFinder search_query: {search_query.model_dump_json()}")

        excluded = set(t.lower() for t in (excluded_titles or []))
        query = search_query or MovieSearchQuery()

        try:
            all_results: list[MovieResult] = []
            seen_ids: set[str] = set()

            if query.has_person_criteria():
                person_movies = self._search_by_persons(query, constraints, limit * 2)
                for movie in person_movies:
                    if movie.id not in seen_ids:
                        seen_ids.add(movie.id)
                        all_results.append(movie)

            discover_movies = self._discover_with_rich_query(query, constraints, limit * 2)
            for movie in discover_movies:
                if movie.id not in seen_ids:
                    seen_ids.add(movie.id)
                    all_results.append(movie)

            if query.text_query:
                text_results = self._client.search_movies(query.text_query, limit=limit)
                for movie in text_results:
                    if movie.id not in seen_ids:
                        seen_ids.add(movie.id)
                        all_results.append(movie)

            results = []
            for movie in all_results:
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

    def _search_by_persons(
        self,
        query: MovieSearchQuery,
        constraints: Constraints,
        limit: int,
    ) -> list[MovieResult]:
        """Search for movies by actors and/or directors.

        Resolves person names to TMDB IDs and uses discover with cast/crew filters.

        Args:
            query: Search query with actor/director names.
            constraints: Hard constraints for filtering.
            limit: Maximum results to return.

        Returns:
            List of movies featuring the specified persons.
        """
        cast_ids: list[int] = []
        crew_ids: list[int] = []

        if query.actors:
            cast_ids = self._client.search_persons(query.actors)
            logger.debug(f"Resolved actors {query.actors} to IDs: {cast_ids}")

        if query.directors:
            crew_ids = self._client.search_persons(query.directors)
            logger.debug(f"Resolved directors {query.directors} to IDs: {crew_ids}")

        if not cast_ids and not crew_ids:
            return []

        return self._client.discover_movies(
            genres=constraints.genres,
            max_runtime=constraints.max_runtime_minutes,
            min_runtime=constraints.min_runtime_minutes,
            year=query.year,
            year_start=query.year_start,
            year_end=query.year_end,
            with_cast=cast_ids if cast_ids else None,
            with_crew=crew_ids if crew_ids else None,
            with_original_language=self._resolve_language(query.language),
            limit=limit,
        )

    def _discover_with_rich_query(
        self,
        query: MovieSearchQuery,
        constraints: Constraints,
        limit: int,
    ) -> list[MovieResult]:
        """Discover movies using all available search criteria.

        Combines genre, year, and keyword filters for comprehensive discovery.

        Args:
            query: Rich search query.
            constraints: Hard constraints.
            limit: Maximum results to return.

        Returns:
            List of discovered movies.
        """
        keyword_ids: list[int] | None = None
        if query.keywords:
            keyword_ids = self._client.search_keywords(query.keywords)
            logger.debug(f"Resolved keywords {query.keywords} to IDs: {keyword_ids}")

        return self._client.discover_movies(
            genres=constraints.genres,
            max_runtime=constraints.max_runtime_minutes,
            min_runtime=constraints.min_runtime_minutes,
            year=query.year,
            year_start=query.year_start,
            year_end=query.year_end,
            with_keywords=keyword_ids if keyword_ids else None,
            with_original_language=self._resolve_language(query.language),
            limit=limit,
        )

    def _resolve_language(self, language: str | None) -> str | None:
        """Resolve language name to ISO 639-1 code.

        Args:
            language: Language name or code.

        Returns:
            ISO 639-1 language code, or None if not resolvable.
        """
        if not language:
            return None

        lang_lower = language.lower()
        if len(lang_lower) == 2:
            return lang_lower

        return LANGUAGE_NAME_TO_CODE.get(lang_lower)
