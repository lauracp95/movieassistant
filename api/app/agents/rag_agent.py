"""RAGAssistantAgent for the Movie Night Assistant."""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import RAG_ASSISTANT_SYSTEM_PROMPT
from app.schemas.domain import RetrievedContext

logger = logging.getLogger(__name__)

__all__ = ["RAGAssistantAgent"]


class RAGAssistantAgent:
    """Answers system questions using retrieved documentation and an LLM."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def answer(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> str:
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
        return f"""## User Question
{query}

## Retrieved Documentation
{context_text}

## Instructions
Answer the user's question based on the retrieved documentation above.
If the documentation doesn't contain relevant information, say so honestly.
Do not make up information that isn't in the documentation."""
