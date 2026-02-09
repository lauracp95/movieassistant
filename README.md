# Movie Night Assistant

A simple chat assistant for planning movie nights, powered by Azure OpenAI via LangChain.

## Architecture

- **Backend**: FastAPI application with `/health` and `/chat` endpoints
- **Frontend**: Streamlit chat interface
- **LLM**: Azure OpenAI (GPT model) via LangChain

## Required Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | ✅ | Azure OpenAI resource endpoint | `https://myresource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | ✅ | Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_API_VERSION` | ✅ | API version | `2024-02-15-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | ✅ | Deployment name (not model name) | `gpt-4o-deployment` |
| `TEMPERATURE` | ❌ | Model temperature (default: 0.7) | `0.7` |
| `MAX_TOKENS` | ❌ | Max response tokens | `1000` |

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
# Edit .env with your actual Azure OpenAI credentials
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

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are some good comedy movies?"}'
```

Response:
```json
{
  "reply": "Great choice! Here are some classic comedies you might enjoy..."
}
```

### Error Responses

- **422**: Invalid input (missing or empty message)
- **500**: Server error (LLM call failed)

## Project Structure

```
.
├── api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app with /health and /chat
│   │   ├── schemas.py        # Pydantic request/response models
│   │   ├── settings.py       # Environment configuration
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── agent.py      # LLM wrapper
│   │       └── prompt.py     # System prompt
│   ├── test/
│   │   ├── conftest.py
│   │   └── test_main.py
│   ├── Dockerfile
│   └── pyproject.toml
├── ui/
│   ├── app/
│   │   └── streamlit_app.py  # Chat interface
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml
└── README.md
```
