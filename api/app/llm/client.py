import logging

from langchain_openai import AzureChatOpenAI

from app.settings import Settings

logger = logging.getLogger(__name__)


def create_chat_model(settings: Settings, temperature: float | None = None) -> AzureChatOpenAI:
    """Create an Azure OpenAI chat model instance.

    Args:
        settings: Application settings with Azure OpenAI configuration.
        temperature: Optional temperature override. Uses settings.temperature if not provided.

    Returns:
        Configured AzureChatOpenAI instance.
    """
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=settings.max_tokens,
    )

