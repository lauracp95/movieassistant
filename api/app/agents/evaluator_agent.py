"""EvaluatorAgent for the Movie Night Assistant."""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.workflow.candidate_selector import detect_constraint_violations
from app.llm.prompts import EVALUATOR_SYSTEM_PROMPT
from app.schemas.domain import DraftRecommendation, EvaluationResult
from app.schemas.input import Constraints

logger = logging.getLogger(__name__)

__all__ = ["EvaluatorAgent"]


class EvaluatorAgent:
    """Judges a :class:`DraftRecommendation` using deterministic checks and an LLM."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm.with_structured_output(EvaluationResult)

    def evaluate(
        self,
        user_message: str,
        constraints: Constraints,
        draft: DraftRecommendation,
        rejected_titles: list[str] | None = None,
    ) -> EvaluationResult:
        violations = detect_constraint_violations(draft, constraints, rejected_titles)
        if violations:
            logger.info(
                f"Evaluator: draft for '{draft.movie.title}' failed "
                f"deterministic pre-check with {len(violations)} violation(s)"
            )
            return EvaluationResult(
                passed=False,
                score=0.0,
                feedback="Draft violates one or more hard constraints.",
                constraint_violations=violations,
                improvement_suggestions=[
                    "pick a different candidate that satisfies the constraints",
                ],
            )

        try:
            result = self._call_llm(
                user_message, constraints, draft, rejected_titles
            )
        except Exception as exc:
            logger.warning(
                f"Evaluator LLM call failed ({exc}); "
                "defaulting to a conservative pass based on deterministic checks"
            )
            return EvaluationResult(
                passed=True,
                score=0.7,
                feedback=(
                    "Evaluator LLM unavailable; draft passed deterministic "
                    "constraint checks."
                ),
                constraint_violations=[],
                improvement_suggestions=[],
            )

        logger.info(
            f"Evaluator: draft for '{draft.movie.title}' scored "
            f"{result.score:.2f}, passed={result.passed}"
        )
        return result

    def _call_llm(
        self,
        user_message: str,
        constraints: Constraints,
        draft: DraftRecommendation,
        rejected_titles: list[str] | None,
    ) -> EvaluationResult:
        human_content = self._build_prompt(
            user_message, constraints, draft, rejected_titles
        )

        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        start = time.time()
        result = self._llm.invoke(messages)
        elapsed = time.time() - start
        logger.info(
            f"Evaluator response ({elapsed:.2f}s): {result.model_dump_json()}"
        )
        return result

    def _build_prompt(
        self,
        user_message: str,
        constraints: Constraints,
        draft: DraftRecommendation,
        rejected_titles: list[str] | None,
    ) -> str:
        constraints_text = self._format_constraints(constraints)
        movie_block = self._format_movie(draft.movie)
        rejected_block = ", ".join(rejected_titles) if rejected_titles else "(none)"

        return (
            f"User request: {user_message}\n\n"
            f"User constraints:\n{constraints_text}\n\n"
            f"Selected movie (the only movie to judge):\n{movie_block}\n\n"
            f"Rejected titles (must not be picked):\n{rejected_block}\n\n"
            f"Recommendation text produced by the writer:\n"
            f'"""\n{draft.recommendation_text}\n"""\n\n'
            "Evaluate the draft now and return the structured JSON verdict."
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

    def _format_movie(self, movie) -> str:
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
