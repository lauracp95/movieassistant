import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import MOVIES_RESPONDER_SYSTEM_PROMPT, SYSTEM_RESPONDER_SYSTEM_PROMPT
from app.schemas.orchestrator import Constraints

logger = logging.getLogger(__name__)


class MoviesResponder:
    """Responder for movie recommendation requests.

    Generates helpful responses based on user message and extracted constraints.
    Currently LLM-only (no external movie database).
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize the movies responder with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm

    def respond(self, user_message: str, constraints: Constraints) -> str:
        """Generate a response for a movie recommendation request.

        Args:
            user_message: The original user message.
            constraints: Extracted constraints (genres, runtime).

        Returns:
            The assistant's text response.

        Raises:
            Exception: If the LLM call fails.
        """
        context_parts = []
        if constraints.genres:
            context_parts.append(f"Genres mentioned: {', '.join(constraints.genres)}")
        if constraints.max_runtime_minutes:
            context_parts.append(f"Max runtime: {constraints.max_runtime_minutes} minutes")
        if constraints.min_runtime_minutes:
            context_parts.append(f"Min runtime: {constraints.min_runtime_minutes} minutes")

        context = "\n".join(context_parts) if context_parts else "No specific constraints detected."

        enhanced_message = f"""User message: {user_message}

Extracted constraints:
{context}

Please respond to the user's movie request, acknowledging any constraints found."""

        messages = [
            SystemMessage(content=MOVIES_RESPONDER_SYSTEM_PROMPT),
            HumanMessage(content=enhanced_message),
        ]

        logger.info(f"MoviesResponder request: {enhanced_message}")
        start_time = time.time()
        response = self._llm.invoke(messages)
        elapsed = time.time() - start_time
        reply = str(response.content)
        logger.info(f"MoviesResponder response ({elapsed:.2f}s): {reply}")
        return reply


class SystemResponder:
    """Responder for system/app questions.

    Answers questions about how the application works, its capabilities, and limitations.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize the system responder with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm

    def respond(self, user_message: str) -> str:
        """Generate a response for a system question.

        Args:
            user_message: The original user message.

        Returns:
            The assistant's text response.

        Raises:
            Exception: If the LLM call fails.
        """
        messages = [
            SystemMessage(content=SYSTEM_RESPONDER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        logger.info(f"SystemResponder request: {user_message}")
        start_time = time.time()
        response = self._llm.invoke(messages)
        elapsed = time.time() - start_time
        reply = str(response.content)
        logger.info(f"SystemResponder response ({elapsed:.2f}s): {reply}")
        return reply
