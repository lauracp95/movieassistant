"""Observability module for the Movie Night Assistant.

This module provides tracing and monitoring capabilities through LangSmith
integration. It centralizes observability configuration and utilities.
"""

from app.observability.langsmith import (
    configure_langsmith,
    get_tracing_status,
    traced_chat,
)

__all__ = [
    "configure_langsmith",
    "get_tracing_status",
    "traced_chat",
]
