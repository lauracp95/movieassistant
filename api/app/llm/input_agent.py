"""InputOrchestratorAgent for the Movie Night Assistant.

This module provides the Phase 2 input agent that handles richer routing
classification with support for movies, rag, and hybrid routes.
"""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.llm.prompts import INPUT_ORCHESTRATOR_SYSTEM_PROMPT
from app.schemas.orchestrator import InputDecision

logger = logging.getLogger(__name__)


class InputOrchestratorAgent:
    """Input Orchestrator Agent that classifies routes and extracts constraints.

    This agent provides Phase 2 routing capabilities:
    - Classifies requests as movies, rag, or hybrid
    - Extracts movie constraints (genres, runtime)
    - Detects clarification needs
    - Generates RAG queries when applicable
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        """Initialize the input orchestrator with a chat model.

        Args:
            llm: Azure OpenAI chat model instance.
        """
        self._llm = llm.with_structured_output(InputDecision)

    def decide(self, user_message: str) -> InputDecision:
        """Analyze a user message and produce a structured routing decision.

        Args:
            user_message: The user's input message.

        Returns:
            InputDecision with route, constraints, clarification info,
            needs_recommendation flag, and optional rag_query.

        Raises:
            Exception: If the LLM call fails or output parsing fails.
        """
        messages = [
            SystemMessage(content=INPUT_ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        logger.info(f"InputOrchestrator request: {user_message}")
        start_time = time.time()
        decision = self._llm.invoke(messages)
        elapsed = time.time() - start_time
        logger.info(
            f"InputOrchestrator response ({elapsed:.2f}s): {decision.model_dump_json()}"
        )

        decision = self._validate_decision(decision)

        return decision

    def _validate_decision(self, decision: InputDecision) -> InputDecision:
        """Validate and normalize the decision output.

        Ensures consistency between route, needs_recommendation, and rag_query.

        Args:
            decision: The raw decision from the LLM.

        Returns:
            Validated and normalized InputDecision.
        """
        if decision.route == "movies":
            decision.needs_recommendation = True
            decision.rag_query = None
        elif decision.route == "rag":
            decision.needs_recommendation = False
        elif decision.route == "hybrid":
            decision.needs_recommendation = True

        return decision
