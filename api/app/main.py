import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from app.agents import (
    EvaluatorAgent,
    InputOrchestratorAgent,
    InMemoryMovieFinderAgent,
    MovieFinderAgent,
    RAGAssistantAgent,
    RecommendationWriterAgent,
    TMDBMovieFinderAgent,
)
from app.guardrails import GuardrailService
from app.routers.routes import cleanup_guardrails, cleanup_workflow, initialize_guardrails, initialize_workflow, router
from app.integrations.tmdb_client import TMDBClient
from langchain_openai import AzureOpenAIEmbeddings

from app.llm import create_chat_model
from app.observability import configure_langsmith, get_tracing_status
from app.rag.retriever import create_retriever
from app.settings import Settings, get_settings
from app.workflow import MovieNightWorkflow

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_tmdb_client: TMDBClient | None = None


def create_movie_finder(settings: Settings) -> MovieFinderAgent:
    """Create the appropriate movie finder based on settings.

    Args:
        settings: Application settings.

    Returns:
        MovieFinderAgent instance (TMDB or in-memory).
    """
    global _tmdb_client
    mode = settings.movie_finder_mode.lower()

    if mode == "inmemory":
        logger.info("Using InMemoryMovieFinderAgent (explicit config)")
        return InMemoryMovieFinderAgent()

    if mode == "tmdb" or (mode == "auto" and settings.tmdb_api_key):
        if not settings.tmdb_api_key:
            logger.warning(
                "TMDB mode requested but no API key; falling back to in-memory finder"
            )
            return InMemoryMovieFinderAgent()

        logger.info("Using TMDBMovieFinderAgent")
        _tmdb_client = TMDBClient(api_key=settings.tmdb_api_key)
        return TMDBMovieFinderAgent(_tmdb_client)

    logger.info("Using InMemoryMovieFinderAgent (no TMDB key)")
    return InMemoryMovieFinderAgent()


def cleanup_tmdb_client() -> None:
    """Close the TMDB client if it exists."""
    global _tmdb_client
    if _tmdb_client is not None:
        _tmdb_client.close()
        _tmdb_client = None
        logger.info("TMDB client closed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize workflow on startup, clean up on shutdown."""
    try:
        settings = get_settings()

        tracing_enabled = configure_langsmith(settings)
        if tracing_enabled:
            status = get_tracing_status()
            logger.info(f"LangSmith tracing active: project={status['project']}")

        input_agent_llm = create_chat_model(settings, temperature=0.0)
        writer_llm = create_chat_model(settings, temperature=0.3)
        evaluator_llm = create_chat_model(settings, temperature=0.0)
        rag_llm = create_chat_model(settings, temperature=0.3)

        input_agent = InputOrchestratorAgent(input_agent_llm)
        movie_finder = create_movie_finder(settings)
        recommendation_writer = RecommendationWriterAgent(writer_llm)
        evaluator = EvaluatorAgent(evaluator_llm)

        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embeddings_deployment,
        )
        rag_retriever = create_retriever(
            embeddings=embeddings,
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
        )
        rag_agent = RAGAssistantAgent(rag_llm)
        logger.info(
            "RAG retriever initialized with %d document chunks",
            len(rag_retriever._documents),
        )

        workflow = MovieNightWorkflow(
            input_agent=input_agent,
            movie_finder=movie_finder,
            rag_retriever=rag_retriever,
            rag_agent=rag_agent,
            recommendation_writer=recommendation_writer,
            evaluator=evaluator,
        )
        initialize_workflow(workflow)
        guardrail_llm = create_chat_model(settings, temperature=0.0)
        guardrail_service = GuardrailService(guardrail_llm, settings)
        initialize_guardrails(guardrail_service)
        logger.info("Guardrail service initialized (enabled=%s)", settings.guardrail_enabled)
        logger.info(
            f"Movie Assistant workflow initialized successfully "
            f"(finder: {type(movie_finder).__name__}, RAG: enabled)"
        )

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit("Failed to start: missing or invalid configuration. Check environment variables.")

    yield

    cleanup_workflow()
    cleanup_guardrails()
    cleanup_tmdb_client()
    logger.info("Movie Assistant workflow cleaned up")


app = FastAPI(
    title="Movie Night Assistant API",
    description="A chat API with intent classification and constraint extraction, powered by Azure OpenAI",
    lifespan=lifespan,
)

app.include_router(router)
