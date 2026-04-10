"""Domain models for the Movie Night Assistant target architecture.

These models represent the internal domain concepts used by the agent workflow:
- MovieResult: A movie retrieved from external sources (e.g., TMDB)
- DraftRecommendation: A candidate recommendation before evaluation
- EvaluationResult: The result of evaluating a draft recommendation
"""

from typing import Literal

from pydantic import BaseModel, Field


class MovieResult(BaseModel):
    """A movie retrieved from an external data source (e.g., TMDB).

    Represents raw movie data before it's processed into a recommendation.
    """

    id: str = Field(
        ...,
        description="Unique identifier from the source (e.g., TMDB ID)",
    )
    title: str = Field(
        ...,
        description="Movie title",
    )
    year: int | None = Field(
        default=None,
        description="Release year",
    )
    genres: list[str] = Field(
        default_factory=list,
        description="List of genre names",
    )
    runtime_minutes: int | None = Field(
        default=None,
        description="Runtime in minutes",
    )
    overview: str | None = Field(
        default=None,
        description="Plot summary or description",
    )
    rating: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Average user rating (0-10 scale)",
    )
    poster_url: str | None = Field(
        default=None,
        description="URL to the movie poster image",
    )
    source: str = Field(
        default="unknown",
        description="Data source identifier (e.g., 'tmdb', 'llm')",
    )


class DraftRecommendation(BaseModel):
    """A candidate movie recommendation before evaluation.

    Contains the movie data plus the generated recommendation text
    that will be evaluated for quality and constraint satisfaction.
    """

    movie: MovieResult = Field(
        ...,
        description="The movie being recommended",
    )
    recommendation_text: str = Field(
        ...,
        description="Generated recommendation text explaining why this movie fits",
    )
    reasoning: str | None = Field(
        default=None,
        description="Internal reasoning for why this movie was selected",
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a draft recommendation.

    Used by the EvaluatorAgent to assess whether a recommendation
    meets quality standards and satisfies user constraints.
    """

    passed: bool = Field(
        ...,
        description="Whether the recommendation passes evaluation",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality score from 0.0 to 1.0",
    )
    feedback: str = Field(
        ...,
        description="Explanation of the evaluation decision",
    )
    constraint_violations: list[str] = Field(
        default_factory=list,
        description="List of violated constraints, if any",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving the recommendation",
    )


class RetrievedContext(BaseModel):
    """A piece of context retrieved from RAG or external sources.

    Used to augment recommendations with additional information.
    """

    content: str = Field(
        ...,
        description="The retrieved text content",
    )
    source: str = Field(
        ...,
        description="Source identifier (e.g., 'rag', 'web')",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance score if available",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata about the source",
    )


class RouteDecision(BaseModel):
    """Routing decision from the InputOrchestratorAgent.

    Determines how a user message should be processed.
    """

    route: Literal["movies", "system", "clarification"] = Field(
        ...,
        description="Target route for processing",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in the routing decision",
    )
    clarification_needed: bool = Field(
        default=False,
        description="Whether clarification is needed from the user",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask if clarification is needed",
    )
