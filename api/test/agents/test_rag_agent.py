"""Unit tests for RAGAssistantAgent."""

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.rag_agent import RAGAssistantAgent
from app.llm.prompts import RAG_ASSISTANT_SYSTEM_PROMPT
from app.schemas.domain import RetrievedContext


def _ctx(
    content: str,
    title: str = "Test Doc",
    source_file: str = "test.md",
    relevance_score: float | None = 0.9,
) -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source="rag",
        relevance_score=relevance_score,
        metadata={"title": title, "source_file": source_file},
    )


class TestRAGAssistantAgentAnswer:
    def test_answer_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="This is the answer based on documentation."
        )
        agent = RAGAssistantAgent(mock_llm)

        answer = agent.answer("How does it work?", [_ctx("Documentation content")])

        mock_llm.invoke.assert_called_once()
        assert answer == "This is the answer based on documentation."

    def test_answer_returns_string_type(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=42)

        agent = RAGAssistantAgent(mock_llm)
        result = agent.answer("query", [])

        assert isinstance(result, str)

    def test_answer_sends_exactly_two_messages(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [])

        messages = mock_llm.invoke.call_args[0][0]
        assert len(messages) == 2

    def test_answer_uses_system_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [])

        messages = mock_llm.invoke.call_args[0][0]
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == RAG_ASSISTANT_SYSTEM_PROMPT

    def test_answer_uses_human_message_for_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [])

        messages = mock_llm.invoke.call_args[0][0]
        assert isinstance(messages[1], HumanMessage)

    def test_answer_includes_query_in_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("What are the known limitations?", [_ctx("Some content")])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "What are the known limitations?" in human_content

    def test_answer_formats_contexts_in_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        contexts = [
            _ctx("First context content", title="First", source_file="first.md", relevance_score=0.9),
            _ctx("Second context content", title="Second", source_file="second.md", relevance_score=0.7),
        ]
        agent.answer("query", contexts)

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "First context content" in human_content
        assert "Second context content" in human_content
        assert "first.md" in human_content
        assert "0.90" in human_content or "0.9" in human_content

    def test_answer_with_empty_contexts(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="No documentation found.")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "No relevant documentation found" in human_content


class TestFormatContexts:
    def test_multiple_contexts_separated_by_divider(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [_ctx("Alpha content"), _ctx("Beta content")])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "---" in human_content

    def test_context_relevance_score_formatted_two_decimals(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [_ctx("Content", relevance_score=0.856)])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "0.86" in human_content

    def test_none_relevance_score_defaults_to_zero(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [_ctx("Content", relevance_score=None)])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "0.00" in human_content

    def test_missing_metadata_uses_unknown_fallback(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        ctx = RetrievedContext(content="Some content", source="rag", metadata={})
        agent.answer("query", [ctx])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "Unknown" in human_content

    def test_contexts_numbered_sequentially(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [_ctx(f"Content {i}") for i in range(3)])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "[Context 1]" in human_content
        assert "[Context 2]" in human_content
        assert "[Context 3]" in human_content

    def test_single_context_has_no_divider(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = RAGAssistantAgent(mock_llm)
        agent.answer("query", [_ctx("Only context")])

        human_content = mock_llm.invoke.call_args[0][0][1].content
        assert "---" not in human_content
