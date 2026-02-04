import pytest
from fastapi.testclient import TestClient

from app.main import app, items


@pytest.fixture(autouse=True)
def clear_items():
    # Ensure tests do not leak state through the global "items" list
    items.clear() # Clean state before the test
    yield
    items.clear() # Clean state after the test


@pytest.fixture()
def client():
    # Provides a TestClient instance for making HTTP requests
    return TestClient(app)


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_create_item_appends_and_returns_full_list(client: TestClient):
    r = client.post("/items", json={"text": "Buy milk", "is_done": False})
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0] == {"text": "Buy milk", "is_done": False}


def test_create_item_defaults_is_done_false(client: TestClient):
    r = client.post("/items", json={"text": "Only text"})
    assert r.status_code == 200

    data = r.json()
    assert len(data) == 1
    assert data[0]["text"] == "Only text"
    assert data[0]["is_done"] is False


def test_create_item_allows_null_text(client: TestClient):
    r = client.post("/items", json={"text": None, "is_done": True})
    assert r.status_code == 200

    data = r.json()
    assert len(data) == 1
    assert data[0] == {"text": None, "is_done": True}


def test_list_items_returns_empty_initially(client: TestClient):
    r = client.get("/items")
    assert r.status_code == 200
    assert r.json() == []


def test_list_items_respects_limit(client: TestClient):
    client.post("/items", json={"text": "A", "is_done": False})
    client.post("/items", json={"text": "B", "is_done": True})
    client.post("/items", json={"text": "C", "is_done": False})

    r = client.get("/items", params={"limit": 2})
    assert r.status_code == 200
    assert r.json() == [
        {"text": "A", "is_done": False},
        {"text": "B", "is_done": True},
    ]


def test_get_item_by_id_success(client: TestClient):
    client.post("/items", json={"text": "First", "is_done": False})
    client.post("/items", json={"text": "Second", "is_done": True})

    r = client.get("/items/1")
    assert r.status_code == 200
    assert r.json() == {"text": "Second", "is_done": True}


def test_get_item_by_id_not_found(client: TestClient):
    client.post("/items", json={"text": "Only one", "is_done": False})

    r = client.get("/items/5")
    assert r.status_code == 404
    assert r.json() == {"detail": "Item 5 not found"}


def test_get_item_by_id_negative_not_found(client: TestClient):
    r = client.get("/items/-1")
    assert r.status_code == 404
    assert r.json() == {"detail": "Item -1 not found"}