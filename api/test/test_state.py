"""Unit tests for the state module."""

from app.llm.state import (
    MAX_MOVIE_SEARCHES,
    MAX_RETRIES,
    PASS_THRESHOLD,
    MovieNightState,
    RouteType,
    create_initial_state,
)
from app.schemas.orchestrator import Constraints


class TestConfigurationConstants:
    def test_max_retries_is_positive(self):
        assert MAX_RETRIES > 0
        assert isinstance(MAX_RETRIES, int)

    def test_pass_threshold_in_valid_range(self):
        assert 0.0 <= PASS_THRESHOLD <= 1.0

    def test_max_movie_searches_is_positive(self):
        assert MAX_MOVIE_SEARCHES > 0
        assert isinstance(MAX_MOVIE_SEARCHES, int)


class TestCreateInitialState:
    def test_creates_state_with_user_message(self):
        state = create_initial_state("Recommend a comedy movie")
        assert state["user_message"] == "Recommend a comedy movie"

    def test_initial_state_has_empty_defaults(self):
        state = create_initial_state("Test message")
        assert state["messages"] == []
        assert state["route"] is None
        assert state["constraints"] is None
        assert state["needs_recommendation"] is False
        assert state["rag_query"] is None
        assert state["candidate_movies"] == []
        assert state["retrieved_contexts"] == []
        assert state["draft_recommendation"] is None
        assert state["evaluation_result"] is None
        assert state["retry_count"] == 0
        assert state["rejected_titles"] == []
        assert state["final_response"] is None
        assert state["error"] is None


class TestMovieNightState:
    def test_state_can_be_updated(self):
        state: MovieNightState = create_initial_state("Test")
        state["route"] = "movies"
        state["constraints"] = Constraints(genres=["comedy"])
        state["retry_count"] = 1

        assert state["route"] == "movies"
        assert state["constraints"].genres == ["comedy"]
        assert state["retry_count"] == 1

    def test_state_supports_all_route_types(self):
        state: MovieNightState = create_initial_state("Test")

        state["route"] = "movies"
        assert state["route"] == "movies"

        state["route"] = "rag"
        assert state["route"] == "rag"

        state["route"] = "hybrid"
        assert state["route"] == "hybrid"

        state["route"] = "clarification"
        assert state["route"] == "clarification"

    def test_state_supports_new_phase2_fields(self):
        state: MovieNightState = create_initial_state("Test")

        state["needs_recommendation"] = True
        assert state["needs_recommendation"] is True

        state["rag_query"] = "How does movie classification work?"
        assert state["rag_query"] == "How does movie classification work?"

    def test_state_can_accumulate_rejected_titles(self):
        state: MovieNightState = create_initial_state("Test")
        state["rejected_titles"].append("Movie A")
        state["rejected_titles"].append("Movie B")

        assert len(state["rejected_titles"]) == 2
        assert "Movie A" in state["rejected_titles"]
        assert "Movie B" in state["rejected_titles"]


class TestRouteType:
    def test_route_type_includes_movies(self):
        route: RouteType = "movies"
        assert route == "movies"

    def test_route_type_includes_rag(self):
        route: RouteType = "rag"
        assert route == "rag"

    def test_route_type_includes_hybrid(self):
        route: RouteType = "hybrid"
        assert route == "hybrid"

    def test_route_type_includes_clarification(self):
        route: RouteType = "clarification"
        assert route == "clarification"
