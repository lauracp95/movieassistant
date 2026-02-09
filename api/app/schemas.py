from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    
    message: str = Field(
        ...,
        min_length=1,
        description="User message to send to the assistant"
    )
    
    @field_validator("message")
    @classmethod
    def message_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v


class ChatResponse(BaseModel):
    """Response body from the /chat endpoint."""
    
    reply: str = Field(
        ...,
        description="Assistant's response to the user message"
    )


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""
    
    status: str = Field(
        default="ok",
        description="Health status of the API"
    )

