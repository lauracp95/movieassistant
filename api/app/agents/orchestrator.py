import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from app.schemas.orchestrator import OrchestratorDecision

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Orchestrator Agent that classifies intent and extracts constraints.

    Uses LangChain's structured output to produce validated Pydantic models.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize the orchestrator with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm.with_structured_output(OrchestratorDecision)

    def decide(self, user_message: str) -> OrchestratorDecision:
        """Analyze a user message and produce a structured decision.

        Args:
            user_message: The user's input message.

        Returns:
            OrchestratorDecision with intent, constraints, and clarification info.

        Raises:
            Exception: If the LLM call fails or output parsing fails.
        """
        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        logger.info(f"Orchestrator request: {user_message}")
        start_time = time.time()
        decision = self._llm.invoke(messages)
        elapsed = time.time() - start_time
        logger.info(f"Orchestrator response ({elapsed:.2f}s): {decision.model_dump_json()}")
        return decision
