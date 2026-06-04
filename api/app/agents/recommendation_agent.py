"""RecommendationWriterAgent for the Movie Night Assistant."""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.workflow.candidate_selector import (
    build_deterministic_recommendation_text,
    build_reasoning,
    select_best_candidate,
)
from app.llm.prompts import RECOMMENDATION_WRITER_SYSTEM_PROMPT
from app.schemas.domain import DraftRecommendation, MovieResult
from app.schemas.input import Constraints

logger = logging.getLogger(__name__)

__all__ = ["RecommendationWriterAgent"]


class RecommendationWriterAgent:
    """Selects a candidate and composes a :class:`DraftRecommendation` using an LLM."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def write(
        self,
        user_message: str,
        constraints: Constraints,
        candidates: list[MovieResult],
        rejected_titles: list[str] | None = None,
    ) -> DraftRecommendation | None:
        logger.info(
            "RecommendationWriter composing draft "
            f"(candidates={len(candidates)}, rejected={len(rejected_titles or [])})"
        )

        movie = select_best_candidate(candidates, constraints, rejected_titles)
        if movie is None:
            logger.info("RecommendationWriter: no candidate survived filtering")
            return None

        reasoning = build_reasoning(movie, constraints)

        try:
            text = self._write_text(user_message, constraints, movie, rejected_titles)
        except Exception as exc:
            logger.warning(
                f"RecommendationWriter LLM call failed ({exc}); "
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
        logger.info(f"RecommendationWriter response ({elapsed:.2f}s): {reply}")

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
        lines: list[str] = []
        if constraints.genres:
            lines.append(f"- genres: {', '.join(constraints.genres)}")
        if constraints.max_runtime_minutes:
            lines.append(f"- max runtime: {constraints.max_runtime_minutes} min")
        if constraints.min_runtime_minutes:
            lines.append(f"- min runtime: {constraints.min_runtime_minutes} min")
        return "\n".join(lines) if lines else "- (none detected)"

    def _format_movie(self, movie: MovieResult) -> str:
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
