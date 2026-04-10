from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.llm.workflow import MovieNightWorkflow
from app.main import app
from app.schemas import Constraints


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_missing_message_field(client: TestClient):
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_empty_message(client: TestClient):
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_whitespace_only_message(client: TestClient):
    r = client.post("/chat", json={"message": "   "})
    assert r.status_code == 422


def test_chat_movies_route():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.get_response.return_value = (
        "Here are some comedy recommendations!",
        "movies",
        Constraints(genres=["comedy"]),
    )

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Recommend a comedy movie"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Here are some comedy recommendations!"
        assert data["route"] == "movies"
        assert data["extracted_constraints"]["genres"] == ["comedy"]
        mock_workflow.get_response.assert_called_once_with("Recommend a comedy movie")


def test_chat_system_route():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.get_response.return_value = (
        "This app uses Azure OpenAI to help with movies.",
        "system",
        Constraints(),
    )

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "How does this app work?"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "This app uses Azure OpenAI to help with movies."
        assert data["route"] == "system"
        mock_workflow.get_response.assert_called_once_with("How does this app work?")


def test_chat_needs_clarification():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.get_response.return_value = (
        "Are you looking for movie recommendations or do you have a question about the app?",
        "clarification",
        Constraints(),
    )

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "help"})

        assert r.status_code == 200
        data = r.json()
        assert "looking for movie recommendations" in data["reply"]
        assert data["route"] == "movies"


def test_chat_workflow_error():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.get_response.side_effect = Exception("Workflow error")

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hello"})

        assert r.status_code == 500
        assert "Failed to generate response" in r.json()["detail"]


def test_chat_workflow_not_initialized():
    with patch("app.api.routes.workflow", None):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hello"})

        assert r.status_code == 500
        assert "not initialized" in r.json()["detail"]
