# Movie Night Assistant

A chat assistant for planning movie nights, powered by Azure OpenAI via LangChain. It runs a LangGraph workflow: an input orchestrator classifies the request, a movie finder pulls candidates from TMDB (or a stub), a recommendation writer drafts grounded text, and an evaluator can reject drafts and drive retries while tracking rejected titles.

## Architecture

- **Backend**: FastAPI application with `/health` and `/chat` endpoints
- **Frontend**: Streamlit chat interface
- **LLM**: Azure OpenAI via LangChain (separate model instances for routing, writing, and evaluation)
- **Workflow**: LangGraph `StateGraph` (`MovieNightWorkflow`) coordinating nodes and conditional edges
- **Input Orchestrator Agent**: Routes to `movies`, `rag`, or `hybrid`; extracts constraints; may ask for clarification
- **Movie Finder Agent**: Retrieves candidate movies from TMDB (or stub data for testing)
- **Recommendation Writer Agent**: Picks a candidate and produces recommendation prose grounded in movie metadata
- **Evaluator Agent**: Scores each draft; on failure the workflow retries (up to `MAX_RETRIES`) and accumulates **rejected titles** so the finder and writer avoid repeating bad picks; exhausted retries yield a safe fallback message
- **Responders**: Movies and system responders for formatting and non-retrieval paths

## How It Works

1. User sends a message to `/chat`
2. **Input Orchestrator** analyzes the message:
   - Chooses route: `movies`, `rag`, `hybrid`, or clarification
   - Extracts constraints (genres, runtime) when relevant
   - Sets `rag_query` for knowledge-style questions
3. **Clarification** ends early with a direct reply.
4. **`rag`** goes straight to **System Responder** (no movie retrieval).
5. **`movies` / `hybrid`** run **Find movies** → **Write recommendation** → **Evaluate** (when an evaluator is configured):
   - The finder can exclude titles in `rejected_titles`
   - If evaluation fails, the draft is cleared, the failed title is added to `rejected_titles`, `retry_count` increments, and the graph loops back to **Write recommendation** while under `MAX_RETRIES`
   - If retries are exhausted (or there is no viable draft), **Respond** uses a polite fallback instead of a low-quality recommendation
6. **Respond** returns final text (draft text when evaluation passed, or formatted candidates / system answer as appropriate)
7. The API returns the reply plus route and extracted constraints

## Required Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | ✅ | Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_ENDPOINT` | ✅ | Azure OpenAI resource endpoint | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | ✅ | Deployment name of the chat model | `gpt-4o` |
| `AZURE_OPENAI_API_VERSION` | ✅ | Azure OpenAI API version | `2024-08-01-preview` |
| `TEMPERATURE` | ❌ | Model temperature (default: 0.7) | `0.7` |
| `MAX_TOKENS` | ❌ | Max response tokens | `1000` |
| `LOG_LEVEL` | ❌ | Logging level (default: INFO) | `DEBUG` |
| `TMDB_API_KEY` | ❌ | TMDB API key for movie data (uses stub if not set) | `abc123...` |
| `MOVIE_FINDER_MODE` | ❌ | Movie finder mode: `auto`, `tmdb`, or `stub` (default: auto) | `auto` |

## Setup Environment Variables

The app loads configuration from a `.env` file at the project root. This file contains secrets and is **gitignored**.

### First-time setup

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

This single `.env` file works for both local development and Docker Compose.

## Run Locally

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- Azure OpenAI resource with a deployed chat model

### Backend (API)

```bash
cd api
uv sync
uv run uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### Backend Tests

```bash
cd api
uv run pytest

# On Windows with Git Bash, use:
uv run python -m pytest
```

### Frontend (UI)

```bash
cd ui

# Optional: Set backend URL (defaults to http://localhost:8000)
export BACKEND_URL="http://localhost:8000"

# Install dependencies and run
uv sync
uv run streamlit run app/streamlit_app.py
```

The UI will be available at http://localhost:8501

## Run with Docker

### Using Docker Compose

```bash
# First time: copy and edit .env file
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

Then run:

```bash
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### Chat

**Movie recommendation request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend a sci-fi movie under 2 hours"}'
```

Response:
```json
{
  "reply": "I see you're looking for a sci-fi movie under 2 hours...",
  "route": "movies",
  "extracted_constraints": {
    "genres": ["sci-fi"],
    "max_runtime_minutes": 120,
    "min_runtime_minutes": null
  }
}
```

**System question:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does this app work?"}'
```

Response:
```json
{
  "reply": "This app uses Azure OpenAI to help you find movies...",
  "route": "rag",
  "extracted_constraints": {
    "genres": [],
    "max_runtime_minutes": null,
    "min_runtime_minutes": null
  }
}
```

With the default startup configuration (`InputOrchestratorAgent`), app questions are classified as `rag`.

### Error Responses

- **422**: Invalid input (missing or empty message)
- **500**: Server error (LLM call failed or agents not initialized)

## Workflow configuration

Retry and pass rules live in `api/app/llm/state.py`:

- `MAX_RETRIES`: maximum evaluation failures before the safe fallback response
- `PASS_THRESHOLD`: minimum evaluator score (combined with the evaluator’s `passed` flag) to accept a draft

The production app wires `LLMEvaluatorAgent` in `api/app/main.py` after the recommendation writer. Tests often use `StubEvaluatorAgent` for deterministic behavior.

## Project Structure

```
.
├── api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app setup and lifespan
│   │   ├── settings.py          # Environment configuration
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py  # Legacy orchestrator (movies vs system)
│   │   │   └── responder.py     # Movies and System responders
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py        # /health and /chat endpoints
│   │   ├── integrations/
│   │   │   ├── __init__.py
│   │   │   └── tmdb_client.py    # TMDB API client
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── client.py         # Azure OpenAI model factory
│   │   │   ├── model_provider.py # ModelProvider class
│   │   │   ├── evaluator_agent.py # Evaluator (stub + LLM, Phase 5)
│   │   │   ├── input_agent.py    # Input orchestrator (movies / rag / hybrid)
│   │   │   ├── movie_finder_agent.py # Movie finder agents (Stub, TMDB)
│   │   │   ├── rag_agent.py      # RAG assistant agent (Phase 6)
│   │   │   ├── recommendation_agent.py # Recommendation writer
│   │   │   ├── prompts.py        # System prompts for all agents
│   │   │   ├── state.py          # MovieNightState, MAX_RETRIES, PASS_THRESHOLD
│   │   │   └── workflow.py       # LangGraph graph, nodes, retry routing
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py         # Document ingestion and chunking
│   │   │   ├── retriever.py      # TF-IDF document retrieval
│   │   │   └── knowledge_base/   # Markdown docs for RAG
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── chat.py           # API request/response models
│   │       ├── domain.py         # Domain models (MovieResult, DraftRecommendation, EvaluationResult, …)
│   │       └── orchestrator.py   # Orchestrator / input decision models
│   ├── test/
│   │   ├── conftest.py
│   │   ├── test_evaluator_agent.py
│   │   ├── test_input_agent.py
│   │   ├── test_main.py
│   │   ├── test_movie_finder.py
│   │   ├── test_rag.py           # RAG retriever, ingester, and agent tests
│   │   ├── test_recommendation_agent.py
│   │   ├── test_state.py
│   │   ├── test_tmdb_client.py
│   │   ├── test_domain.py
│   │   └── test_workflow.py      # LangGraph and end-to-end workflow tests
│   ├── Dockerfile
│   └── pyproject.toml
├── ui/
│   ├── app/
│   │   └── streamlit_app.py     # Chat interface with debug info toggle
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml
└── README.md
```

## Current Limitations

- **Stateless**: No memory between messages
- **No personalized profiles**: No user profiles or watch history tracking
- **Evaluator cost**: Production evaluation adds an extra LLM call per draft attempt (mitigated by deterministic pre-checks for hard constraint violations)
