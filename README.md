# Training Track - API + UI

## Run locally
### API
cd api
uv run uvicorn app.main:app --reload

### API Tests
cd api
uv run pytest

### UI
cd ui
uv run streamlit run app/streamlit_app.py

## Run with Docker
docker compose up --build

## UI available at:
http://localhost:8501
