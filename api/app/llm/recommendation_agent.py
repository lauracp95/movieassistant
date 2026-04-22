"""RecommendationWriterAgent implementations for the Movie Night Assistant.

This module is responsible for the RECOMMENDATION COMPOSITION step of the
pipeline. It is deliberately separated from candidate retrieval
(``MovieFinderAgent``) and from quality evaluation (``EvaluatorAgent``).

Responsibilities:
    1. Use deterministic candidate selection (from candidate_selector module)
    2. Select ONE movie and produce a :class:`DraftRecommendation`
    3. Use the LLM ONLY to write the short natural-language explanation

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

from app.llm.candidate_selector import (
    build_deterministic_recommendation_text,
    build_reasoning,
    filter_candidates,
    prioritize_candidates,
    select_best_candidate,
)
from app.llm.prompts import RECOMMENDATION_WRITER_SYSTEM_PROMPT
from app.schemas.domain import DraftRecommendation, MovieResult
from app.schemas.orchestrator import Constraints

logger = logging.getLogger(__name__)

__all__ = [
    "RecommendationWriterAgent",
    "StubRecommendationWriterAgent",
    "LLMRecommendationWriterAgent",
    "filter_candidates",
    "prioritize_candidates",
    "select_best_candidate",
    "build_reasoning",
]


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
    :func:`build_deterministic_recommendation_text` for the natural-language
    explanation.
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

        text = build_deterministic_recommendation_text(movie, constraints)
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
            text = build_deterministic_recommendation_text(movie, constraints)

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
        human_content = self._build_prompt(
            user_message, constraints, movie, rejected_titles
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
            return build_deterministic_recommendation_text(movie, constraints)

        return reply

    def _build_prompt(
        self,
        user_message: str,
        constraints: Constraints,
        movie: MovieResult,
        rejected_titles: list[str] | None,
    ) -> str:
        """Build the user prompt for the LLM."""
        constraints_text = self._format_constraints(constraints)
        movie_block = self._format_movie(movie)
        rejected_block = ", ".join(rejected_titles) if rejected_titles else "(none)"

        return (
            f"User request: {user_message}\n\n"
            f"User constraints:\n{constraints_text}\n\n"
            f"Selected movie (you MUST only talk about this movie):\n{movie_block}\n\n"
            f"Rejected titles (never mention):\n{rejected_block}\n\n"
            "Write the recommendation text now."
        )

    def _format_constraints(self, constraints: Constraints) -> str:
        """Format constraints for the prompt."""
        lines: list[str] = []
        if constraints.genres:
            lines.append(f"- genres: {', '.join(constraints.genres)}")
        if constraints.max_runtime_minutes:
            lines.append(f"- max runtime: {constraints.max_runtime_minutes} min")
        if constraints.min_runtime_minutes:
            lines.append(f"- min runtime: {constraints.min_runtime_minutes} min")
        return "\n".join(lines) if lines else "- (none detected)"

    def _format_movie(self, movie: MovieResult) -> str:
        """Format movie data for the prompt."""
        lines = [
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
        return "\n".join(lines)
