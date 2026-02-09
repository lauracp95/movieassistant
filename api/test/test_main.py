from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # Provides a TestClient instance for making HTTP requests
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


def test_chat_success():
    with patch("app.main.assistant") as mock_assistant:
        mock_assistant.chat.return_value = "Hello! How can I help you with movie night?"
        
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hi there!"})
        
        assert r.status_code == 200
        assert r.json() == {"reply": "Hello! How can I help you with movie night?"}
        mock_assistant.chat.assert_called_once_with("Hi there!")


def test_chat_llm_error():
    with patch("app.main.assistant") as mock_assistant:
        mock_assistant.chat.side_effect = Exception("API error")
        
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hello"})
        
        assert r.status_code == 500
        assert "Failed to generate response" in r.json()["detail"]


def test_chat_assistant_not_initialized():
    with patch("app.main.assistant", None):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Hello"})
        
        assert r.status_code == 500
        assert "not initialized" in r.json()["detail"]
        