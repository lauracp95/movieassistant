import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from app.agents import OrchestratorAgent, MoviesResponder, SystemResponder
from app.api.routes import router, initialize_agents, cleanup_agents
from app.llm import create_chat_model
from app.settings import get_settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup, clean up on shutdown."""
    try:
        settings = get_settings()

        llm = create_chat_model(settings)
        orchestrator_llm = create_chat_model(settings, temperature=0.0)

        orchestrator = OrchestratorAgent(orchestrator_llm)
        movies_responder = MoviesResponder(llm)
        system_responder = SystemResponder(llm)

        initialize_agents(orchestrator, movies_responder, system_responder)
        logger.info("Movie Assistant agents initialized successfully")

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit("Failed to start: missing or invalid configuration. Check environment variables.")

    # Everything before yield = startup, everything after yield = shutdown.
    yield

    cleanup_agents()
    logger.info("Movie Assistant agents cleaned up")


app = FastAPI(
    title="Movie Night Assistant API",
    description="A chat API with intent classification and constraint extraction, powered by Azure OpenAI",
    lifespan=lifespan,
)

app.include_router(router)
