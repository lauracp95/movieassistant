import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.settings import get_settings
from app.llm.agent import MovieAssistant


# Creates a logger named after this module ("app.main")
logger = logging.getLogger(__name__)

# Module-level global variable that holds the MovieAssistant instance
# "MovieAssistant | None" -> it can be either a MovieAssistant or None: it starts as None and gets initialized in the lifespan function below.
assistant: MovieAssistant | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the LLM assistant on startup, clean up on shutdown."""
    # Modify the module-level variable, not create a new local variable
    global assistant
    
    try:
        settings = get_settings()
        assistant = MovieAssistant(settings)
        logger.info("Movie Assistant initialized successfully")
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit("Failed to start: missing or invalid configuration. Check environment variables.")
    
    # Everything before yield = startup, everything after yield = shutdown.
    yield
    
    assistant = None


app = FastAPI(
    title="Movie Night Assistant API",
    description="A simple chat API backed by Azure OpenAI",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message and return the assistant's reply.
    
    Args:
        request: The chat request containing the user message.
        
    Returns:
        The assistant's response.
        
    Raises:
        HTTPException: If the LLM call fails (500) or input is invalid (422).
    """
    if assistant is None:
        raise HTTPException(
            status_code=500,
            detail="Assistant not initialized"
        )
    
    try:
        reply = assistant.chat(request.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response. Please try again later."
        )
