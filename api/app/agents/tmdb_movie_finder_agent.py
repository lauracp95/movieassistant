"""TMDB-backed movie finder with multi-step search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.agents.movie_finder_agent import MovieFinderAgent
from app.schemas.domain import MovieResult
from app.schemas.input import Constraints, MovieSearchQuery

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


class TMDBMovieFinderAgent(MovieFinderAgent):
    """Retrieves candidates from TMDB using person, discover, and text search."""

    def __init__(self, tmdb_client: "TMDBClient") -> None:
        self._client = tmdb_client

    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
        search_query: MovieSearchQuery | None = None,
    ) -> list[MovieResult]:
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
        if not language:
            return None

        lang_lower = language.lower()
        if len(lang_lower) == 2:
            return lang_lower

        return LANGUAGE_NAME_TO_CODE.get(lang_lower)
