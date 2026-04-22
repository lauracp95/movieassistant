from typing import Literal

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    """Extracted constraints from a movie recommendation request.

    These are hard constraints used for deterministic post-filtering.
    """

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


class MovieSearchQuery(BaseModel):
    """Rich search query extracted from user request for TMDB retrieval.

    This schema captures all search-relevant information from the user's
    natural language request, enabling more targeted movie searches.
    Unlike Constraints (which are for hard filtering), this captures
    semantic search signals that improve initial retrieval quality.
    """

    actors: list[str] = Field(
        default_factory=list,
        description="Actor names mentioned (e.g., 'Tom Hanks', 'Angelina Jolie')",
    )
    directors: list[str] = Field(
        default_factory=list,
        description="Director names mentioned (e.g., 'Christopher Nolan', 'Quentin Tarantino')",
    )
    year: int | None = Field(
        default=None,
        description="Specific release year if mentioned (e.g., 2020)",
    )
    year_start: int | None = Field(
        default=None,
        description="Start of year range for period searches (e.g., 1990 for '90s movies')",
    )
    year_end: int | None = Field(
        default=None,
        description="End of year range for period searches (e.g., 1999 for '90s movies')",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Thematic keywords or plot elements (e.g., 'time travel', 'heist', 'space')",
    )
    mood: str | None = Field(
        default=None,
        description="Mood or tone description (e.g., 'dark', 'lighthearted', 'intense')",
    )
    setting: str | None = Field(
        default=None,
        description="Setting or location (e.g., 'New York', 'space', 'medieval')",
    )
    language: str | None = Field(
        default=None,
        description="Original language preference (e.g., 'Korean', 'French')",
    )
    text_query: str | None = Field(
        default=None,
        description="Free-text search phrase when specific title or franchise is referenced",
    )
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="Things to avoid (e.g., 'violence', 'gore', 'sad ending')",
    )

    def has_person_criteria(self) -> bool:
        """Check if search includes person-based criteria (actors/directors)."""
        return bool(self.actors or self.directors)

    def has_year_criteria(self) -> bool:
        """Check if search includes year/period criteria."""
        return self.year is not None or self.year_start is not None

    def has_keyword_criteria(self) -> bool:
        """Check if search includes keyword/theme criteria."""
        return bool(self.keywords)

    def is_empty(self) -> bool:
        """Check if no search criteria were extracted."""
        return not (
            self.actors
            or self.directors
            or self.year
            or self.year_start
            or self.keywords
            or self.mood
            or self.setting
            or self.language
            or self.text_query
        )


class OrchestratorDecision(BaseModel):
    """Structured output from the basic OrchestratorAgent (movies/system routes only)."""

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
    """Structured output from the InputOrchestratorAgent.

    This schema supports full routing with movies, rag, and hybrid routes.
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
    search_query: MovieSearchQuery = Field(
        default_factory=MovieSearchQuery,
        description="Rich search query for improved TMDB retrieval (actors, directors, year, keywords, etc.)",
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

