from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Constraints, OrchestratorDecision


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
    mock_decision = OrchestratorDecision(
        intent="movies",
        constraints=Constraints(genres=["comedy"]),
        needs_clarification=False,
    )

    with (
        patch("app.api.routes.orchestrator") as mock_orchestrator,
        patch("app.api.routes.movies_responder") as mock_movies,
        patch("app.api.routes.system_responder") as mock_system,
    ):
        mock_orchestrator.decide.return_value = mock_decision
        mock_movies.respond.return_value = "Here are some comedy recommendations!"

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Recommend a comedy movie"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Here are some comedy recommendations!"
        assert data["route"] == "movies"
        assert data["extracted_constraints"]["genres"] == ["comedy"]
        mock_orchestrator.decide.assert_called_once_with("Recommend a comedy movie")
        mock_movies.respond.assert_called_once()
        mock_system.respond.assert_not_called()


def test_chat_system_route():
    mock_decision = OrchestratorDecision(
        intent="system",
        constraints=Constraints(),
        needs_clarification=False,
    )

    with (
        patch("app.api.routes.orchestrator") as mock_orchestrator,
        patch("app.api.routes.movies_responder") as mock_movies,
        patch("app.api.routes.system_responder") as mock_system,
    ):
        mock_orchestrator.decide.return_value = mock_decision
        mock_system.respond.return_value = "This app uses Azure OpenAI to help with movies."

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "How does this app work?"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "This app uses Azure OpenAI to help with movies."
        assert data["route"] == "system"
        mock_orchestrator.decide.assert_called_once_with("How does this app work?")
        mock_system.respond.assert_called_once()
        mock_movies.respond.assert_not_called()


def test_chat_needs_clarification():
    mock_decision = OrchestratorDecision(
        intent="movies",
        constraints=Constraints(),
        needs_clarification=True,
        clarification_question="Are you looking for movie recommendations or do you have a question about the app?",
    )

    with (
        patch("app.api.routes.orchestrator") as mock_orchestrator,
        patch("app.api.routes.movies_responder") as mock_movies,
        patch("app.api.routes.system_responder") as mock_system,
    ):
        mock_orchestrator.decide.return_value = mock_decision

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "help"})

        assert r.status_code == 200
        data = r.json()
        assert "looking for movie recommendations" in data["reply"]
        mock_movies.respond.assert_not_called()
        mock_system.respond.assert_not_called()


def test_chat_llm_error():
    with (
        patch("app.api.routes.orchestrator") as mock_orchestrator,
        patch("app.api.routes.movies_responder"),
        patch("app.api.routes.system_responder"),
    ):
        mock_orchestrator.decide.side_effect = Exception("API error")

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hello"})

        assert r.status_code == 500
        assert "Failed to generate response" in r.json()["detail"]


def test_chat_agents_not_initialized():
    with (
        patch("app.api.routes.orchestrator", None),
        patch("app.api.routes.movies_responder", None),
        patch("app.api.routes.system_responder", None),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hello"})

        assert r.status_code == 500
        assert "not initialized" in r.json()["detail"]
