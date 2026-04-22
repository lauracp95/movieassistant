import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from app.agents import MoviesResponder, SystemResponder
from app.api.routes import cleanup_workflow, initialize_workflow, router
from app.integrations.tmdb_client import TMDBClient
from app.llm import StubMovieFinderAgent, TMDBMovieFinderAgent, create_chat_model
from app.llm.evaluator_agent import LLMEvaluatorAgent
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import MovieFinderAgent
from app.llm.rag_agent import LLMRAGAssistantAgent
from app.llm.recommendation_agent import LLMRecommendationWriterAgent
from app.llm.workflow import MovieNightWorkflow
from app.observability import configure_langsmith, get_tracing_status
from app.rag.retriever import create_retriever
from app.settings import Settings, get_settings

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
        MovieFinderAgent instance (TMDB or Stub).
    """
    global _tmdb_client
    mode = settings.movie_finder_mode.lower()

    if mode == "stub":
        logger.info("Using StubMovieFinderAgent (explicit config)")
        return StubMovieFinderAgent()

    if mode == "tmdb" or (mode == "auto" and settings.tmdb_api_key):
        if not settings.tmdb_api_key:
            logger.warning("TMDB mode requested but no API key; falling back to stub")
            return StubMovieFinderAgent()

        logger.info("Using TMDBMovieFinderAgent")
        _tmdb_client = TMDBClient(api_key=settings.tmdb_api_key)
        return TMDBMovieFinderAgent(_tmdb_client)

    logger.info("Using StubMovieFinderAgent (no TMDB key)")
    return StubMovieFinderAgent()


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

        llm = create_chat_model(settings)
        input_agent_llm = create_chat_model(settings, temperature=0.0)
        writer_llm = create_chat_model(settings, temperature=0.3)
        evaluator_llm = create_chat_model(settings, temperature=0.0)
        rag_llm = create_chat_model(settings, temperature=0.3)

        input_agent = InputOrchestratorAgent(input_agent_llm)
        movies_responder = MoviesResponder(llm)
        system_responder = SystemResponder(llm)
        movie_finder = create_movie_finder(settings)
        recommendation_writer = LLMRecommendationWriterAgent(writer_llm)
        evaluator = LLMEvaluatorAgent(evaluator_llm)

        rag_retriever = create_retriever()
        rag_agent = LLMRAGAssistantAgent(rag_llm)
        logger.info(
            f"RAG retriever initialized with {len(rag_retriever._documents)} documents"
        )

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=movies_responder,
            system_responder=system_responder,
            input_agent=input_agent,
            movie_finder=movie_finder,
            recommendation_writer=recommendation_writer,
            evaluator=evaluator,
            rag_retriever=rag_retriever,
            rag_agent=rag_agent,
        )
        initialize_workflow(workflow)
        logger.info(
            f"Movie Assistant workflow initialized successfully "
            f"(finder: {type(movie_finder).__name__}, RAG: enabled)"
        )

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit("Failed to start: missing or invalid configuration. Check environment variables.")

    yield

    cleanup_workflow()
    cleanup_tmdb_client()
    logger.info("Movie Assistant workflow cleaned up")


app = FastAPI(
    title="Movie Night Assistant API",
    description="A chat API with intent classification and constraint extraction, powered by Azure OpenAI",
    lifespan=lifespan,
)

app.include_router(router)
