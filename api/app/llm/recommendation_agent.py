"""RecommendationWriterAgent implementations for the Movie Night Assistant.

This module is responsible for the RECOMMENDATION COMPOSITION step of the
pipeline. It is deliberately separated from candidate retrieval
(``MovieFinderAgent``) and from quality evaluation (future ``EvaluatorAgent``).

Responsibilities:
    1. Apply deterministic filtering on the candidate list:
       - runtime (max / min) bounds (defense in depth over the finder)
       - exclusion of previously ``rejected_titles``
    2. Apply deterministic prioritization:
       - more constraint-genre matches wins
       - higher rating wins as tiebreaker
       - richer metadata (has overview / has year) wins next
    3. Select ONE movie and produce a :class:`DraftRecommendation`.
    4. Use the LLM ONLY to write the short natural-language explanation.

Implementations:
    - :class:`RecommendationWriterAgent`: abstract base
    - :class:`StubRecommendationWriterAgent`: deterministic, LLM-free
    - :class:`LLMRecommendationWriterAgent`: deterministic selection + LLM text
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import RECOMMENDATION_WRITER_SYSTEM_PROMPT
from app.schemas.domain import DraftRecommendation, MovieResult
from app.schemas.orchestrator import Constraints

logger = logging.getLogger(__name__)


def filter_candidates(
    candidates: list[MovieResult],
    constraints: Constraints,
    rejected_titles: list[str] | None = None,
) -> list[MovieResult]:
    """Deterministically filter candidate movies.

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
        rejected_titles: Titles previously rejected (e.g. by future evaluator).

    Returns:
        A new list of candidates that pass all hard filters.
    """
    excluded = {t.lower() for t in (rejected_titles or [])}

    result: list[MovieResult] = []
    for movie in candidates:
        if movie.title.lower() in excluded:
            continue

        if (
            constraints.max_runtime_minutes is not None
            and movie.runtime_minutes is not None
            and movie.runtime_minutes > constraints.max_runtime_minutes
        ):
            continue

        if (
            constraints.min_runtime_minutes is not None
            and movie.runtime_minutes is not None
            and movie.runtime_minutes < constraints.min_runtime_minutes
        ):
            continue

        result.append(movie)

    return result


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
    """Build a short, deterministic ``reasoning`` field for the draft.

    This is internal reasoning meant for logs / future evaluator consumption,
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


def _build_deterministic_text(movie: MovieResult, constraints: Constraints) -> str:
    """Build a plain-text recommendation grounded strictly in the movie data.

    Used by the stub writer and as a safe fallback if the LLM fails.

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


class RecommendationWriterAgent(ABC):
    """Abstract writer that turns candidates into a :class:`DraftRecommendation`.

    The writer is NOT responsible for retrieval and NOT responsible for
    evaluation. It ONLY selects one candidate and composes the draft.
    """

    @abstractmethod
    def write(
        self,
        user_message: str,
        constraints: Constraints,
        candidates: list[MovieResult],
        rejected_titles: list[str] | None = None,
    ) -> DraftRecommendation | None:
        """Produce a draft recommendation.

        Args:
            user_message: The original user request.
            constraints: Extracted user constraints.
            candidates: Candidate movies retrieved by the finder.
            rejected_titles: Titles the evaluator has previously rejected.

        Returns:
            A valid :class:`DraftRecommendation`, or ``None`` if no
            candidate could be selected (caller should handle gracefully).
        """


class StubRecommendationWriterAgent(RecommendationWriterAgent):
    """Deterministic, LLM-free writer suitable for tests and offline mode.

    Uses :func:`select_best_candidate` for selection and
    :func:`_build_deterministic_text` for the natural-language explanation.
    """

    def write(
        self,
        user_message: str,
        constraints: Constraints,
        candidates: list[MovieResult],
        rejected_titles: list[str] | None = None,
    ) -> DraftRecommendation | None:
        logger.info(
            "StubRecommendationWriter composing draft "
            f"(candidates={len(candidates)}, rejected={len(rejected_titles or [])})"
        )

        movie = select_best_candidate(candidates, constraints, rejected_titles)
        if movie is None:
            logger.info("StubRecommendationWriter: no candidate survived filtering")
            return None

        text = _build_deterministic_text(movie, constraints)
        reasoning = build_reasoning(movie, constraints)

        return DraftRecommendation(
            movie=movie,
            recommendation_text=text,
            reasoning=reasoning,
        )


class LLMRecommendationWriterAgent(RecommendationWriterAgent):
    """Production writer: deterministic selection + LLM-composed text.

    The LLM is used ONLY for writing the short explanation. Movie selection,
    filtering and prioritization remain fully deterministic so the writer's
    behavior is predictable and easy to reason about.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm

    def write(
        self,
        user_message: str,
        constraints: Constraints,
        candidates: list[MovieResult],
        rejected_titles: list[str] | None = None,
    ) -> DraftRecommendation | None:
        logger.info(
            "LLMRecommendationWriter composing draft "
            f"(candidates={len(candidates)}, rejected={len(rejected_titles or [])})"
        )

        movie = select_best_candidate(candidates, constraints, rejected_titles)
        if movie is None:
            logger.info("LLMRecommendationWriter: no candidate survived filtering")
            return None

        reasoning = build_reasoning(movie, constraints)

        try:
            text = self._write_text(user_message, constraints, movie, rejected_titles)
        except Exception as exc:
            logger.warning(
                f"LLMRecommendationWriter LLM call failed ({exc}); "
                "falling back to deterministic text"
            )
            text = _build_deterministic_text(movie, constraints)

        return DraftRecommendation(
            movie=movie,
            recommendation_text=text,
            reasoning=reasoning,
        )

    def _write_text(
        self,
        user_message: str,
        constraints: Constraints,
        movie: MovieResult,
        rejected_titles: list[str] | None,
    ) -> str:
        """Call the LLM to produce the grounded recommendation text."""
        constraint_lines: list[str] = []
        if constraints.genres:
            constraint_lines.append(f"- genres: {', '.join(constraints.genres)}")
        if constraints.max_runtime_minutes:
            constraint_lines.append(
                f"- max runtime: {constraints.max_runtime_minutes} min"
            )
        if constraints.min_runtime_minutes:
            constraint_lines.append(
                f"- min runtime: {constraints.min_runtime_minutes} min"
            )
        constraints_text = (
            "\n".join(constraint_lines) if constraint_lines else "- (none detected)"
        )

        movie_lines = [
            f"- title: {movie.title}",
            f"- year: {movie.year if movie.year is not None else 'unknown'}",
            f"- genres: {', '.join(movie.genres) if movie.genres else 'unknown'}",
            (
                f"- runtime: {movie.runtime_minutes} min"
                if movie.runtime_minutes is not None
                else "- runtime: unknown"
            ),
            (
                f"- rating: {movie.rating:.1f}/10"
                if movie.rating is not None
                else "- rating: unknown"
            ),
            f"- overview: {movie.overview or 'not available'}",
        ]
        movie_block = "\n".join(movie_lines)

        rejected_block = (
            ", ".join(rejected_titles) if rejected_titles else "(none)"
        )

        human_content = (
            f"User request: {user_message}\n\n"
            f"User constraints:\n{constraints_text}\n\n"
            f"Selected movie (you MUST only talk about this movie):\n{movie_block}\n\n"
            f"Rejected titles (never mention):\n{rejected_block}\n\n"
            "Write the recommendation text now."
        )

        messages = [
            SystemMessage(content=RECOMMENDATION_WRITER_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        start = time.time()
        response = self._llm.invoke(messages)
        elapsed = time.time() - start
        reply = str(response.content).strip()
        logger.info(f"LLMRecommendationWriter response ({elapsed:.2f}s): {reply}")

        if not reply:
            logger.warning("LLM returned empty text; falling back to deterministic")
            return _build_deterministic_text(movie, constraints)

        return reply
