# BaseSettings automatically reads values from environment variables
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuration settings for Azure OpenAI integration.
    
    Required environment variables:
        - AZURE_OPENAI_ENDPOINT: Azure OpenAI resource endpoint
        - AZURE_OPENAI_API_KEY: Azure OpenAI API key
        - AZURE_OPENAI_API_VERSION: API version (e.g., 2024-02-15-preview)
        - AZURE_OPENAI_DEPLOYMENT: Deployment name of the chat model
    
    Optional:
        - TEMPERATURE: Model temperature (default: 0.7)
        - MAX_TOKENS: Maximum tokens in response (optional)
    """
    
    # Field(...) = required, no default → app crashes if missing
    azure_openai_endpoint: str = Field(
        ...,
        description="Azure OpenAI resource endpoint (e.g., https://<resource>.openai.azure.com/)"
    )
    azure_openai_api_key: str = Field(
        ...,
        description="Azure OpenAI API key"
    )
    azure_openai_api_version: str = Field(
        ...,
        description="Azure OpenAI API version (e.g., 2024-02-15-preview)"
    )
    azure_openai_deployment: str = Field(
        ...,
        description="Azure OpenAI deployment name for the chat model"
    )
    
    # Optional fields with defaults
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature for response generation"
    )
    # None means "no limit" (let Azure use its default)
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens in the response (optional)"
    )

    # model_config tells Pydantic how to load settings
    model_config = {
        # Look for .env file in parent dir (project root) and current dir
        "env_file": ("../.env", ".env"),
        "env_file_encoding": "utf-8",
        # Ignore extra env vars that don't match our fields
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Load and validate settings from environment variables.
    
    Raises:
        ValidationError: If required environment variables are missing or invalid.
    """
    return Settings()
