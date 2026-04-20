"""EvaluatorAgent implementations for the Movie Night Assistant.

This module is responsible for the EVALUATION step of the pipeline. It takes a
:class:`DraftRecommendation` produced by the writer and returns an
:class:`EvaluationResult` that the workflow uses to decide whether to accept
the draft or retry with a different candidate.

The evaluator is deliberately separated from both retrieval
(:class:`MovieFinderAgent`) and composition
(:class:`RecommendationWriterAgent`). It must not fetch movies, and it must
not rewrite drafts. Its sole responsibility is to judge.

Implementations:
    - :class:`EvaluatorAgent`: abstract base
    - :class:`StubEvaluatorAgent`: deterministic, LLM-free
    - :class:`LLMEvaluatorAgent`: LLM-judged evaluation with structured output

The workflow consults :data:`app.llm.state.PASS_THRESHOLD` to determine the
final pass/fail decision alongside the agent's own ``passed`` flag.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import EVALUATOR_SYSTEM_PROMPT
from app.schemas.domain import DraftRecommendation, EvaluationResult
from app.schemas.orchestrator import Constraints

logger = logging.getLogger(__name__)


def _detect_constraint_violations(
    draft: DraftRecommendation,
    constraints: Constraints,
    rejected_titles: list[str] | None,
) -> list[str]:
    """Deterministically detect hard constraint violations on the draft.

    Used by the stub evaluator and as a safety net in the LLM evaluator.

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


class EvaluatorAgent(ABC):
    """Abstract evaluator that judges a :class:`DraftRecommendation`.

    The evaluator is NOT responsible for retrieval and NOT responsible for
    composition. Given a draft, it produces an :class:`EvaluationResult`.
    """

    @abstractmethod
    def evaluate(
        self,
        user_message: str,
        constraints: Constraints,
        draft: DraftRecommendation,
        rejected_titles: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a draft recommendation.

        Args:
            user_message: The original user request.
            constraints: Extracted user constraints.
            draft: The draft recommendation to judge.
            rejected_titles: Titles rejected in previous retries.

        Returns:
            A valid :class:`EvaluationResult`.
        """


class StubEvaluatorAgent(EvaluatorAgent):
    """Deterministic, LLM-free evaluator suitable for tests and offline mode.

    The stub evaluates purely mechanically:
        - hard constraint violations → fail with score 0.0
        - empty recommendation text → fail with score 0.0
        - otherwise pass with a modest score
    """

    def __init__(self, default_score: float = 0.85) -> None:
        """Initialize the stub evaluator.

        Args:
            default_score: Score assigned to drafts that have no detected
                hard constraint violations.
        """
        self._default_score = default_score

    def evaluate(
        self,
        user_message: str,
        constraints: Constraints,
        draft: DraftRecommendation,
        rejected_titles: list[str] | None = None,
    ) -> EvaluationResult:
        violations = _detect_constraint_violations(
            draft, constraints, rejected_titles
        )

        if violations:
            logger.info(
                f"StubEvaluator: draft for '{draft.movie.title}' failed "
                f"with {len(violations)} violation(s)"
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

        logger.info(
            f"StubEvaluator: draft for '{draft.movie.title}' passed "
            f"(score={self._default_score:.2f})"
        )
        return EvaluationResult(
            passed=True,
            score=self._default_score,
            feedback="Draft satisfies hard constraints.",
            constraint_violations=[],
            improvement_suggestions=[],
        )


class LLMEvaluatorAgent(EvaluatorAgent):
    """Production evaluator using an LLM with structured output.

    A deterministic pre-check still runs first: if the draft violates a hard
    constraint, we fail fast without calling the LLM. Otherwise we delegate
    the quality judgment to the LLM.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm.with_structured_output(EvaluationResult)

    def evaluate(
        self,
        user_message: str,
        constraints: Constraints,
        draft: DraftRecommendation,
        rejected_titles: list[str] | None = None,
    ) -> EvaluationResult:
        violations = _detect_constraint_violations(
            draft, constraints, rejected_titles
        )
        if violations:
            logger.info(
                f"LLMEvaluator: draft for '{draft.movie.title}' failed "
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
                f"LLMEvaluator LLM call failed ({exc}); "
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
            f"LLMEvaluator: draft for '{draft.movie.title}' scored "
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
        """Call the LLM to produce a structured :class:`EvaluationResult`."""
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

        movie = draft.movie
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
            f"Selected movie (the only movie to judge):\n{movie_block}\n\n"
            f"Rejected titles (must not be picked):\n{rejected_block}\n\n"
            f"Recommendation text produced by the writer:\n"
            f"\"\"\"\n{draft.recommendation_text}\n\"\"\"\n\n"
            "Evaluate the draft now and return the structured JSON verdict."
        )

        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        start = time.time()
        result = self._llm.invoke(messages)
        elapsed = time.time() - start
        logger.info(
            f"LLMEvaluator response ({elapsed:.2f}s): "
            f"{result.model_dump_json()}"
        )
        return result
