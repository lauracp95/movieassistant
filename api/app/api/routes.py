import logging

from fastapi import APIRouter, HTTPException

from app.llm.workflow import MovieNightWorkflow
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.schemas.chat import DebugInfo

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


def _build_debug_info(result: dict) -> DebugInfo:
    """Extract debug information from workflow result.

    Args:
        result: The complete workflow state after execution.

    Returns:
        DebugInfo with relevant debugging data.
    """
    retrieved_contexts = []
    for ctx in result.get("retrieved_contexts", []) or []:
        retrieved_contexts.append({
            "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
            "source": ctx.metadata.get("source_file", ctx.source),
            "title": ctx.metadata.get("title", "Unknown"),
            "relevance_score": ctx.relevance_score,
        })

    selected_movie = None
    draft = result.get("draft_recommendation")
    if draft is not None:
        selected_movie = {
            "title": draft.movie.title,
            "year": draft.movie.year,
            "genres": draft.movie.genres,
            "rating": draft.movie.rating,
            "runtime_minutes": draft.movie.runtime_minutes,
        }

    evaluation = None
    eval_result = result.get("evaluation_result")
    if eval_result is not None:
        evaluation = {
            "passed": eval_result.passed,
            "score": eval_result.score,
            "feedback": eval_result.feedback,
            "constraint_violations": eval_result.constraint_violations,
        }

    return DebugInfo(
        rag_query=result.get("rag_query"),
        retrieved_contexts=retrieved_contexts,
        selected_movie=selected_movie,
        evaluation=evaluation,
        retry_count=result.get("retry_count", 0) or 0,
        rejected_titles=result.get("rejected_titles", []) or [],
    )


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

        result = workflow.invoke(request.message)

        final_response = result.get("final_response")
        if not final_response:
            raise RuntimeError("Workflow did not produce a response")

        route = result.get("route")
        constraints = result.get("constraints")

        route_value = None
        if route in ("movies", "rag", "hybrid"):
            route_value = route
        elif route == "clarification":
            route_value = "movies"
        elif route == "system":
            route_value = "rag"

        debug_info = _build_debug_info(result)

        logger.info(f"Chat response generated successfully ({len(final_response)} chars)")

        return ChatResponse(
            reply=final_response,
            route=route_value,
            extracted_constraints=constraints,
            debug=debug_info,
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response. Please try again later.",
        )

