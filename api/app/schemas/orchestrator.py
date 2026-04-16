from typing import Literal

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    """Extracted constraints from a movie recommendation request."""

    genres: list[str] = Field(
        default_factory=list,
        description="List of genres mentioned by the user (e.g., 'sci-fi', 'comedy', 'horror')",
    )
    max_runtime_minutes: int | None = Field(
        default=None,
        description="Maximum runtime in minutes if the user specified a limit",
    )
    min_runtime_minutes: int | None = Field(
        default=None,
        description="Minimum runtime in minutes if the user specified a minimum",
    )


class OrchestratorDecision(BaseModel):
    """Structured output from the Orchestrator Agent (legacy/Phase 1)."""

    intent: Literal["movies", "system"] = Field(
        ...,
        description="Classified intent: 'movies' for recommendation requests, 'system' for app questions",
    )
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Extracted constraints from the user message",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 to 1.0)",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the user message is ambiguous and needs clarification",
    )
    clarification_question: str | None = Field(
        default=None,
        description="A concise clarification question if needs_clarification is True",
    )


class InputDecision(BaseModel):
    """Structured output from the InputOrchestratorAgent (Phase 2).

    This schema supports richer routing with movies, rag, and hybrid routes.
    """

    route: Literal["movies", "rag", "hybrid"] = Field(
        ...,
        description=(
            "Target route: 'movies' for pure movie recommendations, "
            "'rag' for system/knowledge questions, "
            "'hybrid' for requests needing both movie data and RAG context"
        ),
    )
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Extracted movie constraints (genres, runtime) from the user message",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the routing decision (0.0 to 1.0)",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the user message is too ambiguous and needs clarification",
    )
    clarification_question: str | None = Field(
        default=None,
        description="A concise clarification question if needs_clarification is True",
    )
    needs_recommendation: bool = Field(
        default=True,
        description="Whether a movie recommendation should be generated (true for movies/hybrid)",
    )
    rag_query: str | None = Field(
        default=None,
        description=(
            "Query to send to RAG pipeline when route is 'rag' or 'hybrid'. "
            "Should be a well-formed question for knowledge retrieval."
        ),
    )

