"""LangSmith tracing configuration and utilities.

This module centralizes LangSmith/LangChain tracing setup for the Movie Night
Assistant. It provides:

- Configuration initialization from environment variables
- Tracing context managers for request-level traces
- Metadata utilities for enriching traces

LangChain automatically traces LLM calls when LANGCHAIN_TRACING_V2=true.
This module adds request-level tracing with custom metadata for better
debugging and filtering in the LangSmith UI.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from app.settings import Settings

logger = logging.getLogger(__name__)

_tracing_enabled: bool = False


def configure_langsmith(settings: Settings) -> bool:
    """Configure LangSmith tracing from application settings.

    This function sets the required environment variables for LangChain's
    automatic tracing. It should be called once during application startup.

    LangChain reads these environment variables directly:
    - LANGCHAIN_TRACING_V2: Enables tracing when "true"
    - LANGCHAIN_API_KEY: API key for authentication
    - LANGCHAIN_PROJECT: Project name in LangSmith UI
    - LANGCHAIN_ENDPOINT: API endpoint (optional)

    Args:
        settings: Application settings with LangSmith configuration.

    Returns:
        True if tracing was enabled, False otherwise.
    """
    global _tracing_enabled

    if not settings.langsmith_enabled:
        logger.info("LangSmith tracing is disabled (missing API key or not enabled)")
        _tracing_enabled = False
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    _tracing_enabled = True
    logger.info(f"LangSmith tracing enabled for project: {settings.langchain_project}")

    return True


def get_tracing_status() -> dict[str, Any]:
    """Get the current tracing configuration status.

    Returns:
        Dictionary with tracing status information.
    """
    return {
        "enabled": _tracing_enabled,
        "project": os.environ.get("LANGCHAIN_PROJECT", "default"),
        "endpoint": os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    }


def _build_trace_metadata(
    route: str | None = None,
    session_id: str | None = None,
    has_constraints: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build metadata dictionary for a trace.

    Args:
        route: The request route (movies, rag, hybrid).
        session_id: Optional session identifier.
        has_constraints: Whether constraints were extracted.
        **extra: Additional metadata key-value pairs.

    Returns:
        Metadata dictionary for trace tagging.
    """
    metadata = {
        "application": "movie-night-assistant",
    }

    if route:
        metadata["route"] = route
    if session_id:
        metadata["session_id"] = session_id
    if has_constraints:
        metadata["has_constraints"] = True

    metadata.update(extra)
    return metadata


@contextmanager
def traced_chat(
    user_message: str,
    session_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for tracing a chat request.

    Creates a trace context that wraps the entire chat processing flow.
    When LangSmith tracing is enabled, this creates a parent trace that
    groups all LLM calls and workflow steps for a single request.

    The context yields a metadata dictionary that can be updated during
    processing to add information discovered during execution (like route).

    Usage:
        with traced_chat(message, session_id) as trace_meta:
            result = workflow.invoke(message)
            trace_meta["route"] = result.get("route")
            trace_meta["retry_count"] = result.get("retry_count", 0)

    Args:
        user_message: The user's input message.
        session_id: Optional session identifier for grouping conversations.

    Yields:
        A mutable metadata dictionary that will be attached to the trace.
    """
    trace_meta: dict[str, Any] = {
        "session_id": session_id,
        "message_preview": user_message[:100] if user_message else "",
    }

    if not _tracing_enabled:
        yield trace_meta
        return

    try:
        from langsmith import trace

        with trace(
            name="chat_request",
            inputs={"user_message": user_message},
            metadata=_build_trace_metadata(session_id=session_id),
        ) as run:
            yield trace_meta

            run.metadata.update(_build_trace_metadata(
                route=trace_meta.get("route"),
                session_id=session_id,
                has_constraints=trace_meta.get("has_constraints", False),
                retry_count=trace_meta.get("retry_count", 0),
                recommendation_generated=trace_meta.get("recommendation_generated", False),
            ))

    except ImportError:
        logger.warning("langsmith package not available; tracing disabled for this request")
        yield trace_meta
    except Exception as e:
        logger.warning(f"Tracing error (non-fatal): {e}")
        yield trace_meta


def create_trace_tags(
    route: str | None = None,
    evaluation_passed: bool | None = None,
    retry_count: int = 0,
) -> list[str]:
    """Create tags for filtering traces in LangSmith UI.

    Args:
        route: The request route (movies, rag, hybrid).
        evaluation_passed: Whether evaluation passed (if applicable).
        retry_count: Number of retry attempts.

    Returns:
        List of string tags for the trace.
    """
    tags = ["movie-night-assistant"]

    if route:
        tags.append(f"route:{route}")

    if evaluation_passed is not None:
        tags.append("eval:passed" if evaluation_passed else "eval:failed")

    if retry_count > 0:
        tags.append(f"retries:{retry_count}")

    return tags
