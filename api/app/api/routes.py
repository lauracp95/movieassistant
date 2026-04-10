import logging

from fastapi import APIRouter, HTTPException

from app.llm.workflow import MovieNightWorkflow
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

workflow: MovieNightWorkflow | None = None


def initialize_workflow(wf: MovieNightWorkflow) -> None:
    """Initialize the route handlers with a workflow instance.

    Called during application startup.

    Args:
        wf: The compiled MovieNightWorkflow instance.
    """
    global workflow
    workflow = wf


def cleanup_workflow() -> None:
    """Clean up workflow instance during shutdown."""
    global workflow
    workflow = None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message through the LangGraph workflow.

    The workflow orchestrates intent classification, constraint extraction,
    and response generation through a series of graph nodes.

    Args:
        request: The chat request containing the user message.

    Returns:
        The assistant's response with optional route and constraint info.

    Raises:
        HTTPException: If workflow is not initialized (500) or execution fails (500).
    """
    if workflow is None:
        raise HTTPException(
            status_code=500,
            detail="Workflow not initialized",
        )

    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")

        reply, route, constraints = workflow.get_response(request.message)

        route_value = None
        if route in ("movies", "system"):
            route_value = route
        elif route == "clarification":
            route_value = "movies"

        logger.info(f"Chat response generated successfully ({len(reply)} chars)")

        return ChatResponse(
            reply=reply,
            route=route_value,
            extracted_constraints=constraints,
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response. Please try again later.",
        )

