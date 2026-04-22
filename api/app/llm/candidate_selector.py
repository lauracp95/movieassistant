"""Candidate selection, filtering, and constraint validation logic.

This module contains pure functions for deterministic candidate processing:
- Filtering candidates against hard constraints
- Prioritizing candidates based on relevance
- Selecting the best candidate
- Building reasoning and text for recommendations
- Detecting constraint violations (for evaluation)

These functions are separated from the agent classes to enable:
- Independent unit testing of selection logic
- Reuse across different agents (writer, evaluator)
- Clear separation of deterministic vs LLM-based logic
"""

from app.schemas.domain import DraftRecommendation, MovieResult
from app.schemas.orchestrator import Constraints


def filter_candidates(
    candidates: list[MovieResult],
    constraints: Constraints,
    rejected_titles: list[str] | None = None,
) -> list[MovieResult]:
    """Deterministically filter candidate movies against hard constraints.

    This is a defensive filter: the finder may already have filtered by
    genre and runtime, but we enforce the hard constraints again so the
    writer never picks an invalid movie, regardless of source.

    Rules applied:
        - drop any movie whose title is in ``rejected_titles``
          (case-insensitive)
        - drop any movie whose runtime exceeds ``max_runtime_minutes``
        - drop any movie whose runtime is below ``min_runtime_minutes``

    Note:
        Genre mismatch is NOT a hard filter here. If the constraints ask
        for "horror" but only "thriller" candidates came back, we still
        want SOMETHING to show. Genre is used for prioritization only.

    Args:
        candidates: Raw candidate movies.
        constraints: User constraints.
        rejected_titles: Titles previously rejected (e.g. by evaluator).

    Returns:
        A new list of candidates that pass all hard filters.
    """
    excluded = {t.lower() for t in (rejected_titles or [])}

    result: list[MovieResult] = []
    for movie in candidates:
        if movie.title.lower() in excluded:
            continue

        if not _passes_runtime_constraints(movie, constraints):
            continue

        result.append(movie)

    return result


def _passes_runtime_constraints(movie: MovieResult, constraints: Constraints) -> bool:
    """Check if a movie passes runtime constraints.

    Args:
        movie: The movie to check.
        constraints: User constraints with runtime bounds.

    Returns:
        True if the movie passes all runtime constraints.
    """
    if (
        constraints.max_runtime_minutes is not None
        and movie.runtime_minutes is not None
        and movie.runtime_minutes > constraints.max_runtime_minutes
    ):
        return False

    if (
        constraints.min_runtime_minutes is not None
        and movie.runtime_minutes is not None
        and movie.runtime_minutes < constraints.min_runtime_minutes
    ):
        return False

    return True


def prioritize_candidates(
    candidates: list[MovieResult],
    constraints: Constraints,
) -> list[MovieResult]:
    """Deterministically sort candidates from best to worst.

    Ranking key (descending, stable):
        1. Number of genre overlaps with the user's requested genres
        2. Rating (``None`` treated as 0.0)
        3. Has an overview (1 if present, 0 otherwise)
        4. Has a release year (1 if present, 0 otherwise)

    Args:
        candidates: Pre-filtered candidate movies.
        constraints: User constraints driving genre priority.

    Returns:
        A NEW list sorted from best to worst.
    """
    requested_genres = {g.lower() for g in constraints.genres}

    def score(movie: MovieResult) -> tuple[int, float, int, int]:
        movie_genres = {g.lower() for g in movie.genres}
        genre_overlap = len(requested_genres & movie_genres) if requested_genres else 0
        rating = movie.rating if movie.rating is not None else 0.0
        has_overview = 1 if movie.overview else 0
        has_year = 1 if movie.year else 0
        return (genre_overlap, rating, has_overview, has_year)

    return sorted(candidates, key=score, reverse=True)


def select_best_candidate(
    candidates: list[MovieResult],
    constraints: Constraints,
    rejected_titles: list[str] | None = None,
) -> MovieResult | None:
    """Filter and prioritize candidates, then return the best one.

    This is the main entry point for deterministic candidate selection.
    It combines filtering and prioritization into a single call.

    Args:
        candidates: Raw candidates from the finder.
        constraints: User constraints.
        rejected_titles: Titles previously rejected.

    Returns:
        The single best :class:`MovieResult`, or ``None`` if nothing qualifies.
    """
    filtered = filter_candidates(candidates, constraints, rejected_titles)
    if not filtered:
        return None

    prioritized = prioritize_candidates(filtered, constraints)
    return prioritized[0] if prioritized else None


def build_reasoning(movie: MovieResult, constraints: Constraints) -> str:
    """Build a short, deterministic reasoning field for the draft.

    This is internal reasoning meant for logs / evaluator consumption,
    not for the user. It is deliberately mechanical.

    Args:
        movie: The selected movie.
        constraints: User constraints used during selection.

    Returns:
        A short reasoning string describing why this movie was picked.
    """
    parts: list[str] = []

    if constraints.genres:
        requested = {g.lower() for g in constraints.genres}
        movie_genres = {g.lower() for g in movie.genres}
        matched = requested & movie_genres
        if matched:
            parts.append(f"matches genres: {', '.join(sorted(matched))}")
        else:
            parts.append("no exact genre match; closest available candidate")

    if movie.runtime_minutes is not None:
        if constraints.max_runtime_minutes is not None:
            parts.append(
                f"runtime {movie.runtime_minutes}m within "
                f"max {constraints.max_runtime_minutes}m"
            )
        if constraints.min_runtime_minutes is not None:
            parts.append(
                f"runtime {movie.runtime_minutes}m above "
                f"min {constraints.min_runtime_minutes}m"
            )

    if movie.rating is not None:
        parts.append(f"rating {movie.rating:.1f}/10")

    return "; ".join(parts) if parts else "selected as best available candidate"


def build_deterministic_recommendation_text(
    movie: MovieResult,
    constraints: Constraints,
) -> str:
    """Build a plain-text recommendation grounded strictly in the movie data.

    Used by the stub writer and as a safe fallback if the LLM fails.
    This function produces consistent, deterministic output without any
    LLM involvement.

    Args:
        movie: The selected movie.
        constraints: User constraints (used only for light framing).

    Returns:
        A short recommendation text grounded in the movie's data.
    """
    year_str = f" ({movie.year})" if movie.year else ""
    lead = f"I'd suggest **{movie.title}**{year_str}."

    detail_parts: list[str] = []
    if movie.genres:
        detail_parts.append(f"It's a {', '.join(movie.genres[:3]).lower()} pick")
    if movie.runtime_minutes:
        detail_parts.append(f"{movie.runtime_minutes} minutes long")
    if movie.rating is not None:
        detail_parts.append(f"rated {movie.rating:.1f}/10")
    detail = ", ".join(detail_parts) + "." if detail_parts else ""

    overview = f" {movie.overview}" if movie.overview else ""

    fit = ""
    if constraints.genres:
        requested = {g.lower() for g in constraints.genres}
        movie_genres = {g.lower() for g in movie.genres}
        matched = requested & movie_genres
        if matched:
            fit = f" It fits your request for {', '.join(sorted(matched))}."

    pieces = [lead]
    if detail:
        pieces.append(detail)
    if overview:
        pieces.append(overview.strip())
    if fit:
        pieces.append(fit.strip())
    return " ".join(pieces).strip()


def detect_constraint_violations(
    draft: DraftRecommendation,
    constraints: Constraints,
    rejected_titles: list[str] | None,
) -> list[str]:
    """Deterministically detect hard constraint violations on a draft.

    This function checks whether a draft recommendation violates any
    hard constraints. It is used by the evaluator for fast-fail checks
    before invoking the LLM.

    Args:
        draft: The candidate draft recommendation.
        constraints: User constraints.
        rejected_titles: Titles already rejected in previous retries.

    Returns:
        A list of violation strings. Empty if the draft is clean.
    """
    violations: list[str] = []
    movie = draft.movie

    excluded = {t.lower() for t in (rejected_titles or [])}
    if movie.title.lower() in excluded:
        violations.append(f"title '{movie.title}' is in the rejected list")

    if not _passes_runtime_constraints(movie, constraints):
        if (
            constraints.max_runtime_minutes is not None
            and movie.runtime_minutes is not None
            and movie.runtime_minutes > constraints.max_runtime_minutes
        ):
            violations.append(
                f"runtime {movie.runtime_minutes}m exceeds max "
                f"{constraints.max_runtime_minutes}m"
            )
        if (
            constraints.min_runtime_minutes is not None
            and movie.runtime_minutes is not None
            and movie.runtime_minutes < constraints.min_runtime_minutes
        ):
            violations.append(
                f"runtime {movie.runtime_minutes}m is below min "
                f"{constraints.min_runtime_minutes}m"
            )

    if not draft.recommendation_text or not draft.recommendation_text.strip():
        violations.append("recommendation text is empty")

    return violations
