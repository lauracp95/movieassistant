"""RAGAssistantAgent for the Movie Night Assistant.

This module provides the RAG (Retrieval-Augmented Generation) agent that
answers questions about the system using retrieved internal documentation.
"""

import logging
import time
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import RAG_ASSISTANT_SYSTEM_PROMPT
from app.rag.retriever import DocumentRetriever
from app.schemas.domain import RetrievedContext

logger = logging.getLogger(__name__)


class RAGAssistantAgent(ABC):
    """Abstract base class for RAG assistant agents.

    RAG assistant agents answer questions about the system using
    retrieved context from the internal knowledge base.
    """

    @abstractmethod
    def answer(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> str:
        """Generate an answer using retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved contexts from the knowledge base.

        Returns:
            The generated answer grounded in the contexts.
        """
        pass


class StubRAGAssistantAgent(RAGAssistantAgent):
    """Stub implementation for testing without LLM calls.

    Returns a fixed response that includes context information.
    """

    def answer(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> str:
        """Generate a stub answer.

        Args:
            query: The user's question.
            contexts: Retrieved contexts from the knowledge base.

        Returns:
            A stub answer mentioning the retrieved contexts.
        """
        if not contexts:
            return (
                "I don't have specific information about that in my knowledge base. "
                "This is a Movie Night Assistant that helps you discover movies to watch."
            )

        sources = [
            ctx.metadata.get("title", ctx.metadata.get("source_file", "unknown"))
            for ctx in contexts
        ]
        source_list = ", ".join(set(sources))

        return (
            f"Based on my knowledge base ({source_list}), I can help answer "
            f"questions about how the Movie Night Assistant works. "
            f"[Retrieved {len(contexts)} relevant context(s)]"
        )


class LLMRAGAssistantAgent(RAGAssistantAgent):
    """LLM-powered RAG assistant agent.

    Uses Azure OpenAI to generate answers grounded in retrieved
    context from the knowledge base.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize the RAG assistant with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm

    def answer(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> str:
        """Generate an answer using the LLM and retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved contexts from the knowledge base.

        Returns:
            The generated answer grounded in the contexts.
        """
        context_text = self._format_contexts(contexts)

        user_prompt = self._build_user_prompt(query, context_text)

        messages = [
            SystemMessage(content=RAG_ASSISTANT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"RAGAssistant request: {query}")
        start_time = time.time()
        response = self._llm.invoke(messages)
        elapsed = time.time() - start_time
        reply = str(response.content)
        logger.info(f"RAGAssistant response ({elapsed:.2f}s): {reply[:100]}...")

        return reply

    def _format_contexts(self, contexts: list[RetrievedContext]) -> str:
        """Format retrieved contexts for the prompt.

        Args:
            contexts: List of retrieved contexts.

        Returns:
            Formatted context string.
        """
        if not contexts:
            return "No relevant documentation found."

        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            title = ctx.metadata.get("title", "Unknown")
            source = ctx.metadata.get("source_file", "unknown")
            score = ctx.relevance_score or 0.0

            formatted_parts.append(
                f"[Context {i}] Source: {source} | Title: {title} | Relevance: {score:.2f}\n"
                f"{ctx.content}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    def _build_user_prompt(self, query: str, context_text: str) -> str:
        """Build the user prompt with query and context.

        Args:
            query: The user's question.
            context_text: Formatted context text.

        Returns:
            The complete user prompt.
        """
        return f"""## User Question
{query}

## Retrieved Documentation
{context_text}

## Instructions
Answer the user's question based on the retrieved documentation above.
If the documentation doesn't contain relevant information, say so honestly.
Do not make up information that isn't in the documentation."""
