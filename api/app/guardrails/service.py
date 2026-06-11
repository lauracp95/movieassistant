import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

from app.guardrails.patterns import matches_injection_pattern
from app.llm.prompts import GUARDRAIL_SYSTEM_PROMPT
from app.settings import Settings

logger = logging.getLogger(__name__)

__all__ = ["GuardrailResult", "GuardrailService", "LLMGuardrailDecision"]

_INJECTION_REPLY = "I'm sorry, I can't help with that."
_OFF_TOPIC_REPLY = (
    "I'm here to help you find movies to watch or answer questions about how I work. "
    "I can't help with that topic."
)


class LLMGuardrailDecision(BaseModel):
    injection_detected: bool
    off_topic: bool
    reason: str


class GuardrailResult(BaseModel):
    blocked: bool
    reason: Literal["too_long", "injection", "off_topic"] | None = None
    reply: str | None = None


class GuardrailService:
    def __init__(self, llm: AzureChatOpenAI, settings: Settings) -> None:
        self._llm = llm.with_structured_output(LLMGuardrailDecision)
        self._settings = settings

    def check(self, message: str) -> GuardrailResult:
        if not self._settings.guardrail_enabled:
            return GuardrailResult(blocked=False)

        length_result = self._check_length(message)
        if length_result is not None:
            return length_result

        if matches_injection_pattern(message):
            logger.warning("Guardrail: hard injection pattern matched in message")
            return GuardrailResult(blocked=True, reason="injection", reply=_INJECTION_REPLY)

        return self._check_with_llm(message)

    def _check_length(self, message: str) -> GuardrailResult | None:
        limit = self._settings.guardrail_max_message_length
        if len(message) > limit:
            logger.info("Guardrail: message too long (%d chars, limit %d)", len(message), limit)
            return GuardrailResult(
                blocked=True,
                reason="too_long",
                reply=f"Your message is too long. Please keep it under {limit} characters.",
            )
        return None

    def _check_with_llm(self, message: str) -> GuardrailResult:
        try:
            decision: LLMGuardrailDecision = self._llm.invoke([
                SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ])
        except Exception as exc:
            logger.warning("Guardrail LLM check failed (%s); allowing message through", exc)
            return GuardrailResult(blocked=False)

        logger.info("Guardrail LLM decision: %s", decision.model_dump_json())

        if decision.injection_detected:
            return GuardrailResult(blocked=True, reason="injection", reply=_INJECTION_REPLY)
        if decision.off_topic:
            return GuardrailResult(blocked=True, reason="off_topic", reply=_OFF_TOPIC_REPLY)
        return GuardrailResult(blocked=False)
