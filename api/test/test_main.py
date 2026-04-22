from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents import MoviesResponder, SystemResponder
from app.llm.evaluator_agent import StubEvaluatorAgent
from app.llm.input_agent import InputOrchestratorAgent
from app.llm.movie_finder_agent import StubMovieFinderAgent
from app.llm.rag_agent import StubRAGAssistantAgent
from app.llm.recommendation_agent import StubRecommendationWriterAgent
from app.llm.workflow import MovieNightWorkflow
from app.main import app
from app.rag.retriever import create_retriever
from app.schemas import Constraints
from app.schemas.domain import (
    DraftRecommendation,
    EvaluationResult,
    MovieResult,
    RetrievedContext,
)
from app.schemas.orchestrator import InputDecision


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
    mock_workflow.invoke.return_value = {
        "final_response": "Here are some comedy recommendations!",
        "route": "movies",
        "constraints": Constraints(genres=["comedy"]),
        "retrieved_contexts": [],
        "draft_recommendation": None,
        "evaluation_result": None,
        "retry_count": 0,
        "rejected_titles": [],
        "rag_query": None,
    }

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Recommend a comedy movie"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Here are some comedy recommendations!"
        assert data["route"] == "movies"
        assert data["extracted_constraints"]["genres"] == ["comedy"]
        mock_workflow.invoke.assert_called_once_with("Recommend a comedy movie")


def test_chat_rag_route():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.invoke.return_value = {
        "final_response": "This app uses Azure OpenAI to help with movies.",
        "route": "rag",
        "constraints": Constraints(),
        "retrieved_contexts": [],
        "draft_recommendation": None,
        "evaluation_result": None,
        "retry_count": 0,
        "rejected_titles": [],
        "rag_query": "How does this app work?",
    }

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "How does this app work?"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "This app uses Azure OpenAI to help with movies."
        assert data["route"] == "rag"
        mock_workflow.invoke.assert_called_once_with("How does this app work?")


def test_chat_hybrid_route():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.invoke.return_value = {
        "final_response": "Here are horror movies for Halloween with history!",
        "route": "hybrid",
        "constraints": Constraints(genres=["horror"]),
        "retrieved_contexts": [],
        "draft_recommendation": None,
        "evaluation_result": None,
        "retry_count": 0,
        "rejected_titles": [],
        "rag_query": "History of Halloween horror movies",
    }

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Horror movies for Halloween and their history"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Here are horror movies for Halloween with history!"
        assert data["route"] == "hybrid"
        assert data["extracted_constraints"]["genres"] == ["horror"]
        mock_workflow.invoke.assert_called_once_with("Horror movies for Halloween and their history")


def test_chat_system_route_maps_to_rag():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.invoke.return_value = {
        "final_response": "This app uses Azure OpenAI to help with movies.",
        "route": "system",
        "constraints": Constraints(),
        "retrieved_contexts": [],
        "draft_recommendation": None,
        "evaluation_result": None,
        "retry_count": 0,
        "rejected_titles": [],
        "rag_query": None,
    }

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "How does this app work?"})

        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "This app uses Azure OpenAI to help with movies."
        assert data["route"] == "rag"


def test_chat_needs_clarification():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.invoke.return_value = {
        "final_response": "Are you looking for movie recommendations or do you have a question about the app?",
        "route": "clarification",
        "constraints": Constraints(),
        "retrieved_contexts": [],
        "draft_recommendation": None,
        "evaluation_result": None,
        "retry_count": 0,
        "rejected_titles": [],
        "rag_query": None,
    }

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "help"})

        assert r.status_code == 200
        data = r.json()
        assert "looking for movie recommendations" in data["reply"]
        assert data["route"] == "movies"


def test_chat_workflow_error():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.invoke.side_effect = Exception("Workflow error")

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


def test_chat_constraints_with_runtime():
    mock_workflow = MagicMock(spec=MovieNightWorkflow)
    mock_workflow.invoke.return_value = {
        "final_response": "Here's a short comedy for you!",
        "route": "movies",
        "constraints": Constraints(genres=["comedy"], max_runtime_minutes=90),
        "retrieved_contexts": [],
        "draft_recommendation": None,
        "evaluation_result": None,
        "retry_count": 0,
        "rejected_titles": [],
        "rag_query": None,
    }

    with patch("app.api.routes.workflow", mock_workflow):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/chat", json={"message": "Short comedy movie please"})

        assert r.status_code == 200
        data = r.json()
        assert data["route"] == "movies"
        assert data["extracted_constraints"]["genres"] == ["comedy"]
        assert data["extracted_constraints"]["max_runtime_minutes"] == 90


class TestDebugInfo:
    """Tests for the debug info in API responses."""

    def test_chat_returns_debug_info(self):
        movie = MovieResult(
            id="1",
            title="The Matrix",
            year=1999,
            genres=["Sci-Fi", "Action"],
            rating=8.7,
            runtime_minutes=136,
            source="test",
        )
        draft = DraftRecommendation(
            movie=movie,
            recommendation_text="The Matrix is a great sci-fi pick.",
        )
        evaluation = EvaluationResult(
            passed=True,
            score=0.9,
            feedback="Good recommendation",
        )
        contexts = [
            RetrievedContext(
                content="System overview content",
                source="rag",
                relevance_score=0.8,
                metadata={"title": "System Overview", "source_file": "system_overview.md"},
            ),
        ]

        mock_workflow = MagicMock(spec=MovieNightWorkflow)
        mock_workflow.invoke.return_value = {
            "final_response": "The Matrix is a great sci-fi pick.",
            "route": "movies",
            "constraints": Constraints(genres=["sci-fi"]),
            "retrieved_contexts": contexts,
            "draft_recommendation": draft,
            "evaluation_result": evaluation,
            "retry_count": 0,
            "rejected_titles": [],
            "rag_query": None,
        }

        with patch("app.api.routes.workflow", mock_workflow):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/chat", json={"message": "Recommend a sci-fi movie"})

            assert r.status_code == 200
            data = r.json()
            assert data["debug"] is not None

            debug = data["debug"]
            assert debug["selected_movie"]["title"] == "The Matrix"
            assert debug["evaluation"]["passed"] is True
            assert debug["evaluation"]["score"] == 0.9
            assert len(debug["retrieved_contexts"]) == 1
            assert debug["retry_count"] == 0

    def test_chat_debug_with_rejected_titles(self):
        mock_workflow = MagicMock(spec=MovieNightWorkflow)
        mock_workflow.invoke.return_value = {
            "final_response": "Fallback response",
            "route": "movies",
            "constraints": Constraints(genres=["action"]),
            "retrieved_contexts": [],
            "draft_recommendation": None,
            "evaluation_result": EvaluationResult(
                passed=False, score=0.2, feedback="Poor quality"
            ),
            "retry_count": 3,
            "rejected_titles": ["Movie A", "Movie B", "Movie C"],
            "rag_query": None,
        }

        with patch("app.api.routes.workflow", mock_workflow):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/chat", json={"message": "Action movie"})

            assert r.status_code == 200
            data = r.json()
            debug = data["debug"]
            assert debug["retry_count"] == 3
            assert debug["rejected_titles"] == ["Movie A", "Movie B", "Movie C"]

    def test_chat_debug_rag_query(self):
        mock_workflow = MagicMock(spec=MovieNightWorkflow)
        mock_workflow.invoke.return_value = {
            "final_response": "The system works by...",
            "route": "rag",
            "constraints": Constraints(),
            "retrieved_contexts": [],
            "draft_recommendation": None,
            "evaluation_result": None,
            "retry_count": 0,
            "rejected_titles": [],
            "rag_query": "How does the Movie Night Assistant work?",
        }

        with patch("app.api.routes.workflow", mock_workflow):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/chat", json={"message": "How does this work?"})

            assert r.status_code == 200
            data = r.json()
            assert data["debug"]["rag_query"] == "How does the Movie Night Assistant work?"


class TestIntegrationWithStubAgents:
    """Integration tests using stub agents to exercise full workflow paths."""

    @pytest.fixture
    def stub_workflow_movies(self):
        """Create a workflow with stub agents for movies route testing."""
        mock_input_agent = MagicMock(spec=InputOrchestratorAgent)
        mock_input_agent.decide.return_value = InputDecision(
            route="movies",
            constraints=Constraints(genres=["comedy"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query=None,
        )

        mock_movies_responder = MagicMock(spec=MoviesResponder)
        mock_system_responder = MagicMock(spec=SystemResponder)

        return MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=StubMovieFinderAgent(),
            recommendation_writer=StubRecommendationWriterAgent(),
            evaluator=StubEvaluatorAgent(),
        )

    @pytest.fixture
    def stub_workflow_rag(self):
        """Create a workflow with stub agents for RAG route testing."""
        mock_input_agent = MagicMock(spec=InputOrchestratorAgent)
        mock_input_agent.decide.return_value = InputDecision(
            route="rag",
            constraints=Constraints(),
            needs_clarification=False,
            needs_recommendation=False,
            rag_query="How does the system work?",
        )

        mock_movies_responder = MagicMock(spec=MoviesResponder)
        mock_system_responder = MagicMock(spec=SystemResponder)

        rag_retriever = create_retriever()

        return MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=StubMovieFinderAgent(),
            recommendation_writer=StubRecommendationWriterAgent(),
            evaluator=StubEvaluatorAgent(),
            rag_retriever=rag_retriever,
            rag_agent=StubRAGAssistantAgent(),
        )

    @pytest.fixture
    def stub_workflow_hybrid(self):
        """Create a workflow with stub agents for hybrid route testing."""
        mock_input_agent = MagicMock(spec=InputOrchestratorAgent)
        mock_input_agent.decide.return_value = InputDecision(
            route="hybrid",
            constraints=Constraints(genres=["horror"]),
            needs_clarification=False,
            needs_recommendation=True,
            rag_query="History of horror films",
        )

        mock_movies_responder = MagicMock(spec=MoviesResponder)
        mock_system_responder = MagicMock(spec=SystemResponder)

        rag_retriever = create_retriever()

        return MovieNightWorkflow(
            orchestrator=None,
            movies_responder=mock_movies_responder,
            system_responder=mock_system_responder,
            input_agent=mock_input_agent,
            movie_finder=StubMovieFinderAgent(),
            recommendation_writer=StubRecommendationWriterAgent(),
            evaluator=StubEvaluatorAgent(),
            rag_retriever=rag_retriever,
            rag_agent=StubRAGAssistantAgent(),
        )

    def test_movies_route_integration(self, stub_workflow_movies):
        """Test complete movies route with stub agents."""
        with patch("app.api.routes.workflow", stub_workflow_movies):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/chat", json={"message": "Recommend a comedy"})

            assert r.status_code == 200
            data = r.json()
            assert data["route"] == "movies"
            assert data["reply"]
            assert data["extracted_constraints"]["genres"] == ["comedy"]
            assert data["debug"]["selected_movie"] is not None

    def test_rag_route_integration(self, stub_workflow_rag):
        """Test complete RAG route with stub agents."""
        with patch("app.api.routes.workflow", stub_workflow_rag):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/chat", json={"message": "How does this work?"})

            assert r.status_code == 200
            data = r.json()
            assert data["route"] == "rag"
            assert data["reply"]
            assert "knowledge base" in data["reply"].lower()
            assert data["debug"]["rag_query"] == "How does the system work?"

    def test_hybrid_route_integration(self, stub_workflow_hybrid):
        """Test complete hybrid route with stub agents."""
        with patch("app.api.routes.workflow", stub_workflow_hybrid):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/chat", json={"message": "Horror movies and history"})

            assert r.status_code == 200
            data = r.json()
            assert data["route"] == "hybrid"
            assert data["reply"]
            assert data["extracted_constraints"]["genres"] == ["horror"]
            assert data["debug"]["rag_query"] == "History of horror films"

    def test_health_endpoint_always_works(self, stub_workflow_movies):
        """Test health endpoint works regardless of workflow state."""
        with patch("app.api.routes.workflow", stub_workflow_movies):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/health")

            assert r.status_code == 200
            assert r.json() == {"status": "ok"}

        with patch("app.api.routes.workflow", None):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/health")

            assert r.status_code == 200
            assert r.json() == {"status": "ok"}
