"""Abstract contract for candidate movie retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.domain import MovieResult
from app.schemas.input import Constraints, MovieSearchQuery


class MovieFinderAgent(ABC):
    """Retrieve candidate movies from an external source.

    Finders normalize results to :class:`MovieResult` and handle errors
    gracefully. They do not rank or select the final recommendation.
    """

    @abstractmethod
    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
        search_query: MovieSearchQuery | None = None,
    ) -> list[MovieResult]:
        """Find candidate movies matching constraints and optional search query."""
        pass
