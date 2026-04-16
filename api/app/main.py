import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from app.agents import MoviesResponder, SystemResponder
from app.api.routes import cleanup_workflow, initialize_workflow, router
from app.llm import create_chat_model
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.workflow import MovieNightWorkflow
from app.settings import get_settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize workflow on startup, clean up on shutdown."""
    try:
        settings = get_settings()

        llm = create_chat_model(settings)
        input_agent_llm = create_chat_model(settings, temperature=0.0)

        input_agent = InputOrchestratorAgent(input_agent_llm)
        movies_responder = MoviesResponder(llm)
        system_responder = SystemResponder(llm)

        workflow = MovieNightWorkflow(
            orchestrator=None,
            movies_responder=movies_responder,
            system_responder=system_responder,
            input_agent=input_agent,
        )
        initialize_workflow(workflow)
        logger.info("Movie Assistant workflow initialized successfully with InputOrchestratorAgent")

    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit("Failed to start: missing or invalid configuration. Check environment variables.")

    yield

    cleanup_workflow()
    logger.info("Movie Assistant workflow cleaned up")


app = FastAPI(
    title="Movie Night Assistant API",
    description="A chat API with intent classification and constraint extraction, powered by Azure OpenAI",
    lifespan=lifespan,
)

app.include_router(router)
