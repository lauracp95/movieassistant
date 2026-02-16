import logging

from fastapi import APIRouter, HTTPException

from app.agents import OrchestratorAgent, MoviesResponder, SystemResponder
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

orchestrator: OrchestratorAgent | None = None
movies_responder: MoviesResponder | None = None
system_responder: SystemResponder | None = None


def initialize_agents(
    orch: OrchestratorAgent,
    movies: MoviesResponder,
    system: SystemResponder,
) -> None:
    """Initialize the route handlers with agent instances.

    Called during application startup.
    """
    global orchestrator, movies_responder, system_responder
    orchestrator = orch
    movies_responder = movies
    system_responder = system


def cleanup_agents() -> None:
    """Clean up agent instances during shutdown."""
    global orchestrator, movies_responder, system_responder
    orchestrator = None
    movies_responder = None
    system_responder = None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message using the orchestrator and responders.

    The orchestrator classifies the intent and extracts constraints.
    Based on the decision, the appropriate responder generates a reply.

    Args:
        request: The chat request containing the user message.

    Returns:
        The assistant's response with optional route and constraint info.

    Raises:
        HTTPException: If agents are not initialized (500) or LLM call fails (500).
    """
    if orchestrator is None or movies_responder is None or system_responder is None:
        raise HTTPException(
            status_code=500,
            detail="Agents not initialized",
        )

    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")

        decision = orchestrator.decide(request.message)

        logger.debug(
            f"Orchestrator decision: intent={decision.intent}, "
            f"needs_clarification={decision.needs_clarification}, "
            f"constraints={decision.constraints}"
        )

        if decision.needs_clarification:
            reply = decision.clarification_question or "Could you please clarify what you're looking for?"
            return ChatResponse(
                reply=reply,
                route=decision.intent,
                extracted_constraints=decision.constraints,
            )

        if decision.intent == "movies":
            reply = movies_responder.respond(request.message, decision.constraints)
        else:
            reply = system_responder.respond(request.message)

        logger.info(f"Chat response generated successfully ({len(reply)} chars)")

        return ChatResponse(
            reply=reply,
            route=decision.intent,
            extracted_constraints=decision.constraints,
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response. Please try again later.",
        )

