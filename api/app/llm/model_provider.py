"""Centralized Azure OpenAI model provider.

This module provides a single source of truth for creating and configuring
Azure OpenAI chat models. All LLM access in the application should go through
this provider to ensure consistent configuration and easier testing.
"""

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from app.settings import Settings

logger = logging.getLogger(__name__)


class ModelProvider:
    """Centralized provider for Azure OpenAI chat models.

    Encapsulates Azure OpenAI configuration and provides methods to create
    chat model instances with consistent settings. This keeps provider-specific
    code out of agent business logic.

    Example:
        settings = get_settings()
        provider = ModelProvider(settings)
        llm = provider.get_chat_model()
        orchestrator_llm = provider.get_chat_model(temperature=0.0)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the provider with application settings.

        Args:
            settings: Application settings containing Azure OpenAI configuration.
        """
        self._settings = settings
        logger.info(
            f"ModelProvider initialized for deployment: {settings.azure_openai_deployment}"
        )

    @property
    def deployment_name(self) -> str:
        """Get the Azure OpenAI deployment name."""
        return self._settings.azure_openai_deployment

    @property
    def default_temperature(self) -> float:
        """Get the default temperature from settings."""
        return self._settings.temperature

    @property
    def max_tokens(self) -> int | None:
        """Get the max tokens setting."""
        return self._settings.max_tokens

    def get_chat_model(self, temperature: float | None = None) -> BaseChatModel:
        """Create an Azure OpenAI chat model instance.

        Args:
            temperature: Optional temperature override. If not provided,
                uses the default temperature from settings.

        Returns:
            A configured AzureChatOpenAI instance ready for use.
        """
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )

        model = AzureChatOpenAI(
            azure_endpoint=self._settings.azure_openai_endpoint,
            api_key=self._settings.azure_openai_api_key,
            api_version=self._settings.azure_openai_api_version,
            azure_deployment=self._settings.azure_openai_deployment,
            temperature=effective_temperature,
            max_tokens=self._settings.max_tokens,
        )

        logger.debug(
            f"Created chat model: deployment={self._settings.azure_openai_deployment}, "
            f"temperature={effective_temperature}, max_tokens={self._settings.max_tokens}"
        )

        return model


def create_model_provider(settings: Settings) -> ModelProvider:
    """Factory function to create a ModelProvider instance.

    Args:
        settings: Application settings containing Azure OpenAI configuration.

    Returns:
        A configured ModelProvider instance.
    """
    return ModelProvider(settings)


def create_chat_model(settings: Settings, temperature: float | None = None) -> BaseChatModel:
    """Convenience function to create a chat model directly.

    This function provides backwards compatibility with the existing codebase.
    For new code, prefer using ModelProvider directly for better encapsulation.

    Args:
        settings: Application settings with Azure OpenAI configuration.
        temperature: Optional temperature override. Uses settings.temperature if not provided.

    Returns:
        Configured AzureChatOpenAI instance.
    """
    provider = ModelProvider(settings)
    return provider.get_chat_model(temperature)
