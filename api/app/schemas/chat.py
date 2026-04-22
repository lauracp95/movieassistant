from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.orchestrator import Constraints


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        description="User message to send to the assistant",
    )

    @field_validator("message")
    @classmethod
    def message_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v


class DebugInfo(BaseModel):
    """Debug information from workflow execution."""

    rag_query: str | None = Field(
        default=None,
        description="RAG query generated for knowledge retrieval",
    )
    retrieved_contexts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Documents retrieved from the knowledge base",
    )
    selected_movie: dict[str, Any] | None = Field(
        default=None,
        description="Movie selected for recommendation",
    )
    evaluation: dict[str, Any] | None = Field(
        default=None,
        description="Evaluation result for the recommendation",
    )
    retry_count: int = Field(
        default=0,
        description="Number of evaluation retries",
    )
    rejected_titles: list[str] = Field(
        default_factory=list,
        description="Titles rejected during evaluation retries",
    )


class ChatResponse(BaseModel):
    """Response body from the /chat endpoint."""

    reply: str = Field(
        ...,
        description="Assistant's response to the user message",
    )
    route: Literal["movies", "system", "rag", "hybrid"] | None = Field(
        default=None,
        description="The detected intent route (movies, rag, or hybrid)",
    )
    extracted_constraints: Constraints | None = Field(
        default=None,
        description="Extracted constraints from the user message (for debugging)",
    )
    debug: DebugInfo | None = Field(
        default=None,
        description="Debug information from workflow execution (optional)",
    )


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""

    status: str = Field(
        default="ok",
        description="Health status of the API",
    )

