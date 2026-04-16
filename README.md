# Movie Night Assistant

A chat assistant for planning movie nights, powered by Azure OpenAI via LangChain. Features a LangGraph workflow with an Orchestrator Agent that classifies user intent and extracts movie constraints.

## Architecture

- **Backend**: FastAPI application with `/health` and `/chat` endpoints
- **Frontend**: Streamlit chat interface
- **LLM**: Azure OpenAI via LangChain
- **Workflow**: LangGraph StateGraph for orchestration
- **Orchestrator Agent**: Classifies intent (movies vs system) and extracts constraints (genres, runtime)
- **Responders**: Separate response generators for movie requests and system questions

## How It Works

1. User sends a message to `/chat`
2. **Orchestrator Agent** analyzes the message:
   - Classifies intent as "movies" (recommendation request) or "system" (app question)
   - Extracts constraints: genres (sci-fi, comedy, etc.) and runtime limits
   - Determines if clarification is needed
3. Based on the decision, routes to:
   - **Movies Responder**: Handles movie recommendations
   - **System Responder**: Answers questions about the app
4. Returns the response with route and extracted constraints

## Required Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | ✅ | Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_ENDPOINT` | ✅ | Azure OpenAI resource endpoint | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | ✅ | Deployment name of the chat model | `gpt-4o` |
| `AZURE_OPENAI_API_VERSION` | ❌ | API version (default: 2024-08-01-preview) | `2024-08-01-preview` |
| `TEMPERATURE` | ❌ | Model temperature (default: 0.7) | `0.7` |
| `MAX_TOKENS` | ❌ | Max response tokens | `1000` |
| `LOG_LEVEL` | ❌ | Logging level (default: INFO) | `DEBUG` |

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
  "route": "system",
  "extracted_constraints": {
    "genres": [],
    "max_runtime_minutes": null,
    "min_runtime_minutes": null
  }
}
```

### Error Responses

- **422**: Invalid input (missing or empty message)
- **500**: Server error (LLM call failed or agents not initialized)

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
│   │   │   ├── orchestrator.py  # Intent classification and constraint extraction
│   │   │   └── responder.py     # Movies and System responders
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py        # /health and /chat endpoints
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── client.py         # Azure OpenAI model factory
│   │   │   ├── model_provider.py # ModelProvider class
│   │   │   ├── prompts.py        # System prompts for all agents
│   │   │   ├── state.py          # MovieNightState and workflow constants
│   │   │   └── workflow.py       # LangGraph workflow skeleton
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── chat.py           # API request/response models
│   │       ├── domain.py         # Domain models (MovieResult, etc.)
│   │       └── orchestrator.py   # Orchestrator decision models
│   ├── test/
│   │   ├── conftest.py
│   │   └── test_main.py
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

- **No external movie database**: Currently LLM-only, TMDB integration planned for later
- **Stateless**: No memory between messages
- **No RAG**: No retrieval-augmented generation
- **No external tools**: No MCP or API integrations yet
