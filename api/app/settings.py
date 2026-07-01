# BaseSettings automatically reads values from environment variables
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuration settings for Azure OpenAI, external APIs, and observability.

    Required environment variables:
        - AZURE_OPENAI_ENDPOINT: Azure OpenAI resource endpoint
        - AZURE_OPENAI_API_KEY: Azure OpenAI API key
        - AZURE_OPENAI_API_VERSION: API version (e.g., 2024-02-15-preview)
        - AZURE_OPENAI_DEPLOYMENT: Deployment name of the chat model
        - AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: Deployment name for the text-embedding model

    Optional:
        - TEMPERATURE: Model temperature (default: 0.7)
        - MAX_TOKENS: Maximum tokens in response (optional)
        - TMDB_API_KEY: TMDB API key for movie retrieval (optional, uses in-memory catalog if not set)
        - MOVIE_FINDER_MODE: "tmdb", "inmemory", or "auto" (default: auto)
        - CHROMA_PERSIST_DIRECTORY: Directory for ChromaDB persistence (default: None = in-memory)
        - CHROMA_COLLECTION_NAME: Name of the ChromaDB collection (default: knowledge_base)

    Optional (LangSmith tracing; both LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY
    must be set for tracing to activate — see langsmith_enabled):
        - LANGCHAIN_TRACING_V2: Enable tracing (default: false)
        - LANGCHAIN_API_KEY: LangSmith API key
        - LANGCHAIN_PROJECT: Project name in LangSmith UI (default: movie-night-assistant)
        - LANGCHAIN_ENDPOINT: LangSmith API endpoint (default: https://api.smith.langchain.com)
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
    azure_openai_embeddings_deployment: str | None = Field(
        default=None,
        description="Azure OpenAI deployment name for the text-embedding model (used by the RAG retriever). "
        "If not set, RAG features are disabled."
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
    
    # TMDB integration (optional)
    tmdb_api_key: str | None = Field(
        default=None,
        description="TMDB API key for movie retrieval (if not set, uses in-memory finder)"
    )
    movie_finder_mode: str = Field(
        default="auto",
        description="Movie finder mode: 'tmdb', 'inmemory', or 'auto' (auto-detect based on API key)"
    )

    # ChromaDB / RAG retriever (optional)
    chroma_persist_directory: str | None = Field(
        default=None,
        description="Directory for ChromaDB persistence. None = ephemeral in-memory collection."
    )
    chroma_collection_name: str = Field(
        default="knowledge_base",
        description="Name of the ChromaDB collection used by the RAG retriever"
    )

    # Guardrails
    guardrail_max_message_length: int = Field(
        default=2000,
        gt=0,
        description="Maximum allowed message length in characters before the guardrail blocks the request",
    )
    guardrail_enabled: bool = Field(
        default=True,
        description="Enable guardrail checks (length, injection, topic) before workflow invocation",
    )

    # LangSmith observability (optional)
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing for LLM calls and workflow runs"
    )
    langchain_api_key: str | None = Field(
        default=None,
        description="LangSmith API key for tracing"
    )
    langchain_project: str = Field(
        default="movie-night-assistant",
        description="LangSmith project name for organizing traces"
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint"
    )

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is properly configured and enabled."""
        return self.langchain_tracing_v2 and self.langchain_api_key is not None

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
