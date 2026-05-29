import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import SYSTEM_RESPONDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class SystemResponder:
    """Responder for system/app questions.

    Answers questions about how the application works, its capabilities, and limitations.
    Used as a fallback when the RAG pipeline is not configured or for unexpected routes.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    def respond(self, user_message: str) -> str:
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
