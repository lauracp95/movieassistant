"""In-memory movie finder for tests and local development."""

from __future__ import annotations

import logging

from app.agents.movie_finder_agent import MovieFinderAgent
from app.schemas.domain import MovieResult
from app.schemas.input import Constraints, MovieSearchQuery

logger = logging.getLogger(__name__)


class StubMovieFinderAgent(MovieFinderAgent):
    """Returns predictable movie data without external API calls."""

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
        self._movies = custom_movies if custom_movies is not None else self.STUB_MOVIES

    def find_movies(
        self,
        constraints: Constraints,
        limit: int = 10,
        excluded_titles: list[str] | None = None,
        search_query: MovieSearchQuery | None = None,
    ) -> list[MovieResult]:
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
