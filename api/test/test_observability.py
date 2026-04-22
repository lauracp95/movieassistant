"""Tests for the observability module.

These tests verify that LangSmith tracing:
- Can be configured and enabled via settings
- Gracefully degrades when disabled or unconfigured
- Does not crash the application when langsmith is unavailable
- Properly enriches traces with metadata
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.observability.langsmith import (
    _build_trace_metadata,
    configure_langsmith,
    create_trace_tags,
    get_tracing_status,
    traced_chat,
)


class TestConfigureLangsmith:
    """Tests for configure_langsmith function."""

    def test_disabled_when_tracing_false(self):
        """Tracing should be disabled when langchain_tracing_v2 is False."""
        settings = MagicMock()
        settings.langsmith_enabled = False

        result = configure_langsmith(settings)

        assert result is False

    def test_disabled_when_api_key_missing(self):
        """Tracing should be disabled when API key is missing."""
        settings = MagicMock()
        settings.langsmith_enabled = False
        settings.langchain_api_key = None

        result = configure_langsmith(settings)

        assert result is False

    def test_enabled_when_properly_configured(self):
        """Tracing should be enabled with valid configuration."""
        settings = MagicMock()
        settings.langsmith_enabled = True
        settings.langchain_api_key = "test-api-key"
        settings.langchain_project = "test-project"
        settings.langchain_endpoint = "https://api.smith.langchain.com"

        with patch.dict(os.environ, {}, clear=False):
            result = configure_langsmith(settings)

            assert result is True
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == "test-api-key"
            assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"


class TestGetTracingStatus:
    """Tests for get_tracing_status function."""

    def test_returns_status_dict(self):
        """Should return a dictionary with status information."""
        status = get_tracing_status()

        assert isinstance(status, dict)
        assert "enabled" in status
        assert "project" in status
        assert "endpoint" in status


class TestTracedChat:
    """Tests for traced_chat context manager."""

    def test_yields_metadata_dict_when_disabled(self):
        """Should yield metadata dict even when tracing is disabled."""
        with traced_chat("test message", session_id="test-session") as meta:
            assert isinstance(meta, dict)
            assert meta["session_id"] == "test-session"
            assert "message_preview" in meta

    def test_allows_metadata_enrichment(self):
        """Should allow updating metadata during execution."""
        with traced_chat("test message") as meta:
            meta["route"] = "movies"
            meta["has_constraints"] = True
            meta["custom_field"] = "custom_value"

            assert meta["route"] == "movies"
            assert meta["has_constraints"] is True
            assert meta["custom_field"] == "custom_value"

    def test_handles_long_messages(self):
        """Should truncate long messages in preview."""
        long_message = "x" * 500
        with traced_chat(long_message) as meta:
            assert len(meta["message_preview"]) == 100

    def test_handles_empty_message(self):
        """Should handle empty messages gracefully."""
        with traced_chat("") as meta:
            assert meta["message_preview"] == ""


class TestBuildTraceMetadata:
    """Tests for _build_trace_metadata helper."""

    def test_always_includes_application(self):
        """Should always include application identifier."""
        metadata = _build_trace_metadata()
        assert metadata["application"] == "movie-night-assistant"

    def test_includes_route_when_provided(self):
        """Should include route when provided."""
        metadata = _build_trace_metadata(route="movies")
        assert metadata["route"] == "movies"

    def test_includes_session_id_when_provided(self):
        """Should include session_id when provided."""
        metadata = _build_trace_metadata(session_id="session-123")
        assert metadata["session_id"] == "session-123"

    def test_includes_extra_kwargs(self):
        """Should include any extra keyword arguments."""
        metadata = _build_trace_metadata(custom_field="value")
        assert metadata["custom_field"] == "value"


class TestCreateTraceTags:
    """Tests for create_trace_tags function."""

    def test_always_includes_app_tag(self):
        """Should always include the application tag."""
        tags = create_trace_tags()
        assert "movie-night-assistant" in tags

    def test_includes_route_tag(self):
        """Should include route tag when provided."""
        tags = create_trace_tags(route="hybrid")
        assert "route:hybrid" in tags

    def test_includes_eval_passed_tag(self):
        """Should include eval:passed tag when evaluation passed."""
        tags = create_trace_tags(evaluation_passed=True)
        assert "eval:passed" in tags

    def test_includes_eval_failed_tag(self):
        """Should include eval:failed tag when evaluation failed."""
        tags = create_trace_tags(evaluation_passed=False)
        assert "eval:failed" in tags

    def test_includes_retry_tag(self):
        """Should include retry tag when retries occurred."""
        tags = create_trace_tags(retry_count=2)
        assert "retries:2" in tags

    def test_no_retry_tag_when_zero(self):
        """Should not include retry tag when retry_count is 0."""
        tags = create_trace_tags(retry_count=0)
        assert not any(tag.startswith("retries:") for tag in tags)
