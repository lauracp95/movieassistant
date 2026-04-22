"""Unit tests for the EvaluatorAgent (Phase 5)."""

from unittest.mock import MagicMock

import pytest
from langchain_openai import AzureChatOpenAI

from app.llm.candidate_selector import detect_constraint_violations
from app.llm.evaluator_agent import (
    LLMEvaluatorAgent,
    StubEvaluatorAgent,
)
from app.schemas.domain import DraftRecommendation, EvaluationResult, MovieResult
from app.schemas.orchestrator import Constraints


def _movie(
    id_: str,
    title: str,
    genres: list[str] | None = None,
    runtime_minutes: int | None = None,
    rating: float | None = None,
    overview: str | None = None,
    year: int | None = None,
) -> MovieResult:
    return MovieResult(
        id=id_,
        title=title,
        genres=genres or [],
        runtime_minutes=runtime_minutes,
        rating=rating,
        overview=overview,
        year=year,
        source="test",
    )


def _draft(
    movie: MovieResult,
    text: str = "A short and grounded recommendation.",
) -> DraftRecommendation:
    return DraftRecommendation(
        movie=movie,
        recommendation_text=text,
        reasoning="deterministic",
    )


class TestDetectConstraintViolations:
    def test_no_violations_for_clean_draft(self):
        draft = _draft(_movie("1", "Clean", runtime_minutes=100))
        violations = detect_constraint_violations(
            draft, Constraints(max_runtime_minutes=120), rejected_titles=[]
        )
        assert violations == []

    def test_flags_rejected_title(self):
        draft = _draft(_movie("1", "Bad Movie", runtime_minutes=100))
        violations = detect_constraint_violations(
            draft, Constraints(), rejected_titles=["Bad Movie"]
        )
        assert any("rejected" in v.lower() for v in violations)

    def test_rejected_title_is_case_insensitive(self):
        draft = _draft(_movie("1", "The Matrix", runtime_minutes=100))
        violations = detect_constraint_violations(
            draft, Constraints(), rejected_titles=["the matrix"]
        )
        assert any("rejected" in v.lower() for v in violations)

    def test_flags_runtime_over_max(self):
        draft = _draft(_movie("1", "Long", runtime_minutes=200))
        violations = detect_constraint_violations(
            draft, Constraints(max_runtime_minutes=120), rejected_titles=[]
        )
        assert any("exceeds max" in v for v in violations)

    def test_flags_runtime_below_min(self):
        draft = _draft(_movie("1", "Short", runtime_minutes=60))
        violations = detect_constraint_violations(
            draft, Constraints(min_runtime_minutes=90), rejected_titles=[]
        )
        assert any("below min" in v for v in violations)

    def test_flags_empty_recommendation_text(self):
        draft = _draft(_movie("1", "Empty"), text="   ")
        violations = detect_constraint_violations(
            draft, Constraints(), rejected_titles=[]
        )
        assert any("empty" in v for v in violations)


class TestStubEvaluator:
    def test_passes_clean_draft(self):
        evaluator = StubEvaluatorAgent(default_score=0.9)
        draft = _draft(_movie("1", "Clean", runtime_minutes=100))

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(max_runtime_minutes=120),
            draft=draft,
            rejected_titles=[],
        )

        assert isinstance(result, EvaluationResult)
        assert result.passed is True
        assert result.score == 0.9
        assert result.constraint_violations == []
        assert result.improvement_suggestions == []

    def test_fails_draft_with_runtime_violation(self):
        evaluator = StubEvaluatorAgent()
        draft = _draft(_movie("1", "Long", runtime_minutes=200))

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(max_runtime_minutes=120),
            draft=draft,
            rejected_titles=[],
        )

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.constraint_violations) >= 1
        assert result.improvement_suggestions  # non-empty

    def test_fails_draft_with_rejected_title(self):
        evaluator = StubEvaluatorAgent()
        draft = _draft(_movie("1", "Already Rejected"))

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(),
            draft=draft,
            rejected_titles=["Already Rejected"],
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_fails_draft_with_empty_text(self):
        evaluator = StubEvaluatorAgent()
        draft = DraftRecommendation(
            movie=_movie("1", "Clean"),
            recommendation_text="",
        )

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(),
            draft=draft,
            rejected_titles=[],
        )

        assert result.passed is False

    def test_handles_none_rejected_titles(self):
        evaluator = StubEvaluatorAgent()
        draft = _draft(_movie("1", "Clean"))

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(),
            draft=draft,
            rejected_titles=None,
        )

        assert result.passed is True


class TestLLMEvaluator:
    def test_short_circuits_on_hard_violation_without_calling_llm(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        structured = MagicMock()
        llm.with_structured_output.return_value = structured

        evaluator = LLMEvaluatorAgent(llm)

        draft = _draft(_movie("1", "Long", runtime_minutes=300))
        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(max_runtime_minutes=120),
            draft=draft,
            rejected_titles=[],
        )

        assert result.passed is False
        assert result.score == 0.0
        structured.invoke.assert_not_called()

    def test_uses_llm_when_no_hard_violations(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        structured = MagicMock()
        structured.invoke.return_value = EvaluationResult(
            passed=True,
            score=0.88,
            feedback="Great pick.",
            constraint_violations=[],
            improvement_suggestions=[],
        )
        llm.with_structured_output.return_value = structured

        evaluator = LLMEvaluatorAgent(llm)
        draft = _draft(_movie("1", "Clean", runtime_minutes=100))

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(max_runtime_minutes=120),
            draft=draft,
            rejected_titles=[],
        )

        assert result.passed is True
        assert result.score == 0.88
        structured.invoke.assert_called_once()

    def test_llm_failure_falls_back_to_conservative_pass(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        structured = MagicMock()
        structured.invoke.side_effect = RuntimeError("boom")
        llm.with_structured_output.return_value = structured

        evaluator = LLMEvaluatorAgent(llm)
        draft = _draft(_movie("1", "Clean", runtime_minutes=100))

        result = evaluator.evaluate(
            user_message="recommend",
            constraints=Constraints(),
            draft=draft,
            rejected_titles=[],
        )

        assert result.passed is True
        assert result.score == 0.7
        assert "deterministic" in result.feedback.lower()

    def test_llm_receives_prompt_with_user_request_and_draft_text(self):
        llm = MagicMock(spec=AzureChatOpenAI)
        structured = MagicMock()
        structured.invoke.return_value = EvaluationResult(
            passed=True,
            score=0.8,
            feedback="ok",
        )
        llm.with_structured_output.return_value = structured

        evaluator = LLMEvaluatorAgent(llm)
        draft = _draft(
            _movie("1", "The Matrix", genres=["sci-fi"], rating=8.7),
            text="Watch The Matrix, a grounded sci-fi pick.",
        )

        evaluator.evaluate(
            user_message="sci-fi please",
            constraints=Constraints(genres=["sci-fi"]),
            draft=draft,
            rejected_titles=["Inception"],
        )

        messages = structured.invoke.call_args[0][0]
        assert len(messages) == 2
        system_content = messages[0].content
        human_content = messages[1].content
        assert "EvaluatorAgent" in system_content or "evaluate" in system_content.lower()
        assert "sci-fi please" in human_content
        assert "The Matrix" in human_content
        assert "Watch The Matrix" in human_content
        assert "Inception" in human_content


class TestThresholdIntegration:
    """Verify pass/fail integrates with PASS_THRESHOLD at workflow level."""

    def test_score_above_threshold_with_passed_flag_accepted(self):
        from app.llm.state import PASS_THRESHOLD

        result = EvaluationResult(
            passed=True, score=0.9, feedback="ok"
        )
        assert result.passed and result.score >= PASS_THRESHOLD

    def test_score_below_threshold_rejected(self):
        from app.llm.state import PASS_THRESHOLD

        result = EvaluationResult(
            passed=True, score=0.5, feedback="weak"
        )
        assert not (result.passed and result.score >= PASS_THRESHOLD)

    def test_passed_false_rejected_even_with_high_score(self):
        from app.llm.state import PASS_THRESHOLD

        result = EvaluationResult(
            passed=False, score=0.95, feedback="nope"
        )
        assert not (result.passed and result.score >= PASS_THRESHOLD)
