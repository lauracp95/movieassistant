"""LLM client package for the Movie Night Assistant.

This package provides the Azure OpenAI client factory. All agent
implementations have moved to :mod:`app.agents`.
"""

from app.llm.client import create_chat_model

__all__ = [
    "create_chat_model",
]
