# Movie Night Assistant

A chat assistant for planning movie nights, powered by Azure OpenAI via LangChain. It runs a LangGraph workflow: an input orchestrator classifies the request, a movie finder pulls candidates from TMDB (or an in-memory catalog), a recommendation writer drafts grounded text, and an evaluator can reject drafts and drive retries while tracking rejected titles.

## Architecture

- **Backend**: FastAPI application with `/health` and `/chat` endpoints
- **Frontend**: Streamlit chat interface with debug info panel
- **LLM**: Azure OpenAI via LangChain (separate model instances for routing, writing, evaluation, RAG, and guardrails)
- **Workflow**: LangGraph `StateGraph` (`MovieNightWorkflow`) coordinating nodes and conditional edges
- **Guardrail Service**: Pre-workflow safety layer that blocks messages exceeding the length limit, matching injection patterns, or identified as off-topic by LLM classification
- **Input Orchestrator Agent**: Routes to `movies`, `rag`, or `hybrid`; extracts constraints and rich search signals (actors, directors, year, keywords, mood, setting, language); may ask for clarification
- **Movie Finder Agent**: Retrieves candidate movies from TMDB, or an in-memory catalog when no API key is configured (`InMemoryMovieFinderAgent`)
- **Recommendation Writer Agent**: Selects a candidate and produces recommendation prose grounded in movie metadata
- **Evaluator Agent**: Validates drafts against constraints and quality criteria; on failure the workflow retries (up to `MAX_RETRIES`) and accumulates **rejected titles** so the writer avoids repeating bad picks; exhausted retries yield a safe fallback message
- **RAG Assistant Agent**: Answers system questions using retrieved documentation from the knowledge base
- **Document Retriever**: ChromaDB vector store with Azure OpenAI embeddings for semantic search over markdown knowledge base files
- **LangSmith** (optional): Traces LLM calls and `/chat` workflow runs when `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` are set

## How It Works

1. User sends a message to `/chat`
2. **Guardrail Service** inspects the message:
   - Blocks if message exceeds `GUARDRAIL_MAX_MESSAGE_LENGTH` characters
   - Blocks if a hard injection pattern matches
   - Blocks if the LLM classifies the message as a prompt injection or off-topic
3. **Input Orchestrator** analyzes the message:
   - Chooses route: `movies`, `rag`, `hybrid`, or clarification
   - Extracts constraints (genres, runtime) when relevant
   - Extracts rich search signals: actors, directors, year/year range, keywords, mood, setting, language
   - Sets `rag_query` for knowledge-style questions
4. **Clarification** ends early with a direct reply.
5. **`rag`** retrieves relevant documents from the knowledge base and generates a grounded answer via the RAG Assistant Agent.
6. **`movies`** runs **Find movies** → **Write recommendation** → **Evaluate** (when an evaluator is configured):
   - The finder can exclude titles in `rejected_titles`
   - If evaluation fails, the draft is cleared, the failed title is added to `rejected_titles`, `retry_count` increments, and the graph loops back to **Write recommendation** while under `MAX_RETRIES`
   - If retries are exhausted (or there is no viable draft), **Respond** uses a polite fallback instead of a low-quality recommendation
7. **`hybrid`** combines both flows: **Find movies** → **RAG retrieve** → **Write recommendation** → **Evaluate** → **Respond**
8. **Respond** returns final text (draft text when evaluation passed, RAG answer, or formatted candidates as appropriate)
9. The API returns the reply plus route and extracted constraints

## Required Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | ✅ | Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_ENDPOINT` | ✅ | Azure OpenAI resource endpoint | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | ✅ | Deployment name of the chat model | `gpt-4o` |
| `AZURE_OPENAI_API_VERSION` | ✅ | Azure OpenAI API version | `2024-08-01-preview` |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | ✅ | Deployment name of the text-embedding model (used by RAG retriever) | `text-embedding-ada-002` |
| `TEMPERATURE` | ❌ | Base model temperature (default: 0.7; individual agents use fixed values: 0.0 for input/evaluator/guardrails, 0.3 for writer/RAG) | `0.7` |
| `MAX_TOKENS` | ❌ | Max response tokens | `1000` |
| `LOG_LEVEL` | ❌ | Logging level (default: INFO) | `DEBUG` |
| `TMDB_API_KEY` | ❌ | TMDB API key for movie data (uses in-memory catalog if not set) | `abc123...` |
| `MOVIE_FINDER_MODE` | ❌ | Movie finder mode: `auto`, `tmdb`, or `inmemory` (default: auto) | `auto` |
| `CHROMA_PERSIST_DIRECTORY` | ❌ | Directory for ChromaDB persistence (default: in-memory, ephemeral) | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | ❌ | ChromaDB collection name (default: `knowledge_base`) | `knowledge_base` |
| `GUARDRAIL_ENABLED` | ❌ | Enable guardrail checks before workflow (default: true) | `true` |
| `GUARDRAIL_MAX_MESSAGE_LENGTH` | ❌ | Maximum allowed message length in characters (default: 2000) | `2000` |
| `LANGCHAIN_TRACING_V2` | ❌ | Enable LangSmith tracing (default: false) | `true` |
| `LANGCHAIN_API_KEY` | ❌ | LangSmith API key (required when tracing is enabled) | `lsv2_...` |
| `LANGCHAIN_PROJECT` | ❌ | LangSmith project name (default: `movie-night-assistant`) | `movie-night-assistant` |
| `LANGCHAIN_ENDPOINT` | ❌ | LangSmith API endpoint | `https://api.smith.langchain.com` |

Tracing is active only when `LANGCHAIN_TRACING_V2=true` **and** `LANGCHAIN_API_KEY` is set (see `api/app/settings.py` and `api/app/observability/langsmith.py`).

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
- Azure OpenAI resource with a deployed chat model and an embeddings model

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

Tests do not call Azure OpenAI or TMDB by default:

- **Writer, evaluator, and RAG agents**: `unittest.mock.MagicMock` stands in for the LangChain chat model (`api/test/conftest.py` fixtures `recommendation_writer`, `evaluator`, `rag_agent`).
- **Input orchestrator**: mocked at the agent interface where the workflow is under test.
- **Movie finder**: `InMemoryMovieFinderAgent` provides a fixed in-memory catalog (same implementation used when `MOVIE_FINDER_MODE=inmemory` or no `TMDB_API_KEY` in production).

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

Retry and pass rules live in `api/app/workflow/state.py`:

- `MAX_RETRIES`: maximum evaluation failures before the safe fallback response
- `PASS_THRESHOLD`: minimum evaluator score (combined with the evaluator's `passed` flag) to accept a draft

The production app wires `RecommendationWriterAgent`, `EvaluatorAgent`, and `RAGAssistantAgent` in `api/app/main.py` (each takes a mocked LLM in tests). Only the movie finder has a non-LLM implementation (`InMemoryMovieFinderAgent` when TMDB is unavailable).

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
│   │   │   ├── input_agent.py              # Route classifier (movies/rag/hybrid)
│   │   │   ├── movie_finder_agent.py       # MovieFinderAgent protocol
│   │   │   ├── in_memory_movie_finder_agent.py  # In-memory catalog (no TMDB key)
│   │   │   ├── tmdb_movie_finder_agent.py  # TMDB-backed finder
│   │   │   ├── recommendation_agent.py     # LLM recommendation writer
│   │   │   ├── evaluator_agent.py          # LLM draft validator
│   │   │   └── rag_agent.py                # LLM RAG assistant
│   │   ├── guardrails/
│   │   │   ├── __init__.py
│   │   │   ├── service.py       # GuardrailService: length, injection, off-topic checks
│   │   │   └── patterns.py      # Compiled injection regex patterns
│   │   ├── integrations/
│   │   │   └── tmdb_client.py   # TMDB API client
│   │   ├── knowledge_base/      # Markdown docs ingested into ChromaDB for RAG
│   │   │   ├── system_overview.md
│   │   │   ├── recommendation_rules.md
│   │   │   ├── evaluation_logic.md
│   │   │   ├── data_sources.md
│   │   │   ├── routing_logic.md
│   │   │   └── known_limitations.md
│   │   ├── llm/
│   │   │   ├── client.py        # Azure OpenAI model factory
│   │   │   └── prompts.py       # System prompts for all agents
│   │   ├── observability/
│   │   │   └── langsmith.py     # LangSmith configuration and tracing helpers
│   │   ├── rag/
│   │   │   ├── ingest.py        # Document ingestion and chunking
│   │   │   ├── chroma_store.py  # ChromaDB vector store management
│   │   │   └── retriever.py     # Semantic document retrieval (top_k=3)
│   │   ├── routers/
│   │   │   └── routes.py        # /health and /chat endpoints
│   │   ├── schemas/
│   │   │   ├── chat.py          # API request/response models (ChatRequest, ChatResponse)
│   │   │   ├── domain.py        # Domain models (MovieResult, DraftRecommendation, etc.)
│   │   │   └── input.py         # Input models (Constraints, MovieSearchQuery, InputDecision)
│   │   └── workflow/
│   │       ├── graph_builder.py # LangGraph MovieNightWorkflow
│   │       ├── nodes.py         # Graph node implementations
│   │       ├── routing.py       # Conditional edges
│   │       ├── state.py         # MovieNightState, MAX_RETRIES, PASS_THRESHOLD
│   │       ├── candidate_selector.py # Deterministic selection and constraint checks
│   │       └── formatters.py    # Response formatting helpers
│   ├── test/
│   │   ├── conftest.py
│   │   ├── test_main.py              # API endpoint integration tests
│   │   ├── test_settings.py          # Settings validation tests
│   │   ├── agents/
│   │   │   ├── test_input_agent.py
│   │   │   ├── test_movie_finder.py
│   │   │   ├── test_recommendation_agent.py
│   │   │   ├── test_evaluator_agent.py
│   │   │   └── test_rag_agent.py
│   │   ├── guardrails/
│   │   │   ├── test_guardrail_service.py
│   │   │   └── test_patterns.py
│   │   ├── integrations/
│   │   │   ├── test_tmdb_client.py
│   │   │   └── test_tmdb_client_extended.py
│   │   ├── observability/
│   │   │   └── test_observability.py
│   │   ├── rag/
│   │   │   └── test_rag.py
│   │   ├── schemas/
│   │   │   ├── test_domain.py
│   │   │   └── test_search_query.py
│   │   └── workflow/
│   │       ├── test_state.py
│   │       ├── test_workflow_nodes.py
│   │       ├── test_workflow_routing.py
│   │       ├── test_workflow_evaluator.py
│   │       ├── test_workflow_rag.py
│   │       └── test_workflow_integration.py
│   ├── Dockerfile
│   └── pyproject.toml
├── ui/
│   ├── app/
│   │   └── streamlit_app.py     # Chat interface with debug info panel
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml
├── .env.example
└── README.md
```

## Current Limitations

- **Stateless**: No memory between messages
- **No personalized profiles**: No user profiles or watch history tracking
- **Evaluator cost**: Production evaluation adds an extra LLM call per draft attempt (mitigated by deterministic pre-checks for hard constraint violations)
- **Guardrail cost**: When `GUARDRAIL_ENABLED=true`, every request performs an LLM classification call in addition to the workflow
