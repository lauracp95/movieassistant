"""Unit tests for RAG retriever, ingest, and agent components."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from app.llm.rag_agent import (
    LLMRAGAssistantAgent,
    RAGAssistantAgent,
    StubRAGAssistantAgent,
)
from app.rag.ingest import (
    DEFAULT_CHUNK_SIZE,
    DocumentIngester,
    KnowledgeDocument,
)
from app.rag.retriever import (
    DocumentRetriever,
    create_retriever,
)
from app.schemas.domain import RetrievedContext


class TestDocumentIngester:
    def test_load_documents_from_empty_directory(self):
        with TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            docs = ingester.load_documents()
            assert docs == []

    def test_load_documents_from_nonexistent_directory(self):
        ingester = DocumentIngester(
            knowledge_base_path=Path("/nonexistent/path")
        )
        docs = ingester.load_documents()
        assert docs == []

    def test_load_single_markdown_file(self):
        with TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "test.md"
            md_file.write_text("# Test Title\n\nSome content here.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            docs = ingester.load_documents()

            assert len(docs) >= 1
            assert docs[0].title == "Test Title"
            assert docs[0].source == "test.md"
            assert "Some content here" in docs[0].content

    def test_load_multiple_markdown_files(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc1.md").write_text("# Doc One\n\nFirst document.")
            (Path(tmpdir) / "doc2.md").write_text("# Doc Two\n\nSecond document.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            docs = ingester.load_documents()

            assert len(docs) >= 2
            titles = [d.title for d in docs]
            assert "Doc One" in titles
            assert "Doc Two" in titles

    def test_extract_title_from_h1_heading(self):
        with TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "test.md"
            md_file.write_text("# My Document Title\n\nBody text.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            docs = ingester.load_documents()

            assert docs[0].title == "My Document Title"

    def test_extract_title_fallback_to_filename(self):
        with TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "my_document.md"
            md_file.write_text("No heading here, just text.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            docs = ingester.load_documents()

            assert docs[0].title == "My Document"

    def test_chunk_large_document(self):
        with TemporaryDirectory() as tmpdir:
            large_content = "# Large Doc\n\n" + ("This is a paragraph. " * 100 + "\n\n") * 10
            md_file = Path(tmpdir) / "large.md"
            md_file.write_text(large_content)

            ingester = DocumentIngester(
                knowledge_base_path=Path(tmpdir),
                chunk_size=500,
                chunk_overlap=50,
            )
            docs = ingester.load_documents()

            assert len(docs) > 1
            for doc in docs:
                assert doc.metadata.get("chunk_index") is not None

    def test_small_document_not_chunked(self):
        with TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "small.md"
            md_file.write_text("# Small\n\nJust a tiny document.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            docs = ingester.load_documents()

            assert len(docs) == 1
            assert docs[0].metadata.get("total_chunks") == 1

    def test_documents_property_after_loading(self):
        with TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "test.md"
            md_file.write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            assert ingester.documents == []

            ingester.load_documents()
            assert len(ingester.documents) >= 1


class TestDocumentRetriever:
    def test_retrieve_returns_empty_list_when_no_documents(self):
        with TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester)

            results = retriever.retrieve("any query")
            assert results == []

    def test_retrieve_finds_relevant_documents(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "movies.md").write_text(
                "# Movies\n\n" +
                "This comprehensive document covers movies recommendations viewers cinema " +
                "films entertainment watching suggestions picks choices selections."
            )
            (Path(tmpdir) / "system.md").write_text(
                "# System\n\n" +
                "This detailed document explains system architecture design software " +
                "engineering infrastructure deployment scalability performance."
            )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester, top_k=2, min_score=0.0)
            retriever.initialize()

            results = retriever.retrieve("movies recommendations viewers cinema films")

            assert len(results) >= 1
            movie_results = [r for r in results if "movies" in r.content.lower()]
            assert len(movie_results) >= 1

    def test_retrieve_respects_top_k(self):
        with TemporaryDirectory() as tmpdir:
            for i in range(5):
                (Path(tmpdir) / f"doc{i}.md").write_text(
                    f"# Document {i}\n\nThis is document number {i} about topics."
                )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester, top_k=2)
            retriever.initialize()

            results = retriever.retrieve("document topics")

            assert len(results) <= 2

    def test_retrieve_returns_retrieved_context_objects(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text(
                "# Test\n\nContent for testing retrieval."
            )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester)
            retriever.initialize()

            results = retriever.retrieve("testing retrieval")

            assert len(results) >= 1
            for r in results:
                assert isinstance(r, RetrievedContext)
                assert r.source == "rag"
                assert r.relevance_score is not None
                assert "title" in r.metadata

    def test_retrieve_empty_query_returns_empty(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester)
            retriever.initialize()

            results = retriever.retrieve("the a an is")
            assert results == []

    def test_retrieve_all_returns_all_documents(self):
        with TemporaryDirectory() as tmpdir:
            for i in range(3):
                (Path(tmpdir) / f"doc{i}.md").write_text(f"# Doc {i}\n\nContent {i}.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester)
            retriever.initialize()

            results = retriever.retrieve_all()

            assert len(results) == 3
            for r in results:
                assert r.relevance_score == 1.0

    def test_auto_initialize_on_first_retrieve(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(ingester=ingester)

            assert not retriever._initialized
            retriever.retrieve("anything")
            assert retriever._initialized

    def test_create_retriever_factory_function(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = create_retriever(ingester=ingester)

            assert retriever._initialized
            assert len(retriever._documents) >= 1


class TestStubRAGAssistantAgent:
    def test_answer_with_no_contexts(self):
        agent = StubRAGAssistantAgent()
        answer = agent.answer("How does this work?", contexts=[])

        assert "don't have specific information" in answer.lower()

    def test_answer_with_contexts(self):
        agent = StubRAGAssistantAgent()
        contexts = [
            RetrievedContext(
                content="System overview content",
                source="rag",
                relevance_score=0.9,
                metadata={"title": "System Overview"},
            ),
        ]

        answer = agent.answer("How does this work?", contexts)

        assert "knowledge base" in answer.lower()
        assert "System Overview" in answer

    def test_answer_includes_context_count(self):
        agent = StubRAGAssistantAgent()
        contexts = [
            RetrievedContext(
                content="Content 1",
                source="rag",
                metadata={"title": "Doc 1"},
            ),
            RetrievedContext(
                content="Content 2",
                source="rag",
                metadata={"title": "Doc 2"},
            ),
        ]

        answer = agent.answer("question", contexts)
        assert "2" in answer


class TestLLMRAGAssistantAgent:
    def test_answer_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="This is the answer based on documentation."
        )

        agent = LLMRAGAssistantAgent(mock_llm)
        contexts = [
            RetrievedContext(
                content="Documentation content",
                source="rag",
                relevance_score=0.8,
                metadata={"title": "Test Doc", "source_file": "test.md"},
            ),
        ]

        answer = agent.answer("How does it work?", contexts)

        mock_llm.invoke.assert_called_once()
        assert answer == "This is the answer based on documentation."

    def test_answer_formats_contexts_in_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        agent = LLMRAGAssistantAgent(mock_llm)
        contexts = [
            RetrievedContext(
                content="First context content",
                source="rag",
                relevance_score=0.9,
                metadata={"title": "First", "source_file": "first.md"},
            ),
            RetrievedContext(
                content="Second context content",
                source="rag",
                relevance_score=0.7,
                metadata={"title": "Second", "source_file": "second.md"},
            ),
        ]

        agent.answer("query", contexts)

        call_args = mock_llm.invoke.call_args[0][0]
        user_message = call_args[1].content

        assert "First context content" in user_message
        assert "Second context content" in user_message
        assert "first.md" in user_message
        assert "0.90" in user_message or "0.9" in user_message

    def test_answer_with_empty_contexts(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="No documentation found.")

        agent = LLMRAGAssistantAgent(mock_llm)
        answer = agent.answer("query", [])

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        user_message = call_args[1].content

        assert "No relevant documentation found" in user_message


class TestKnowledgeBaseFiles:
    """Test that the actual knowledge base files exist and can be loaded."""

    def test_knowledge_base_files_exist(self):
        from app.rag.ingest import DEFAULT_KNOWLEDGE_BASE_PATH

        required_files = [
            "system_overview.md",
            "recommendation_rules.md",
            "known_limitations.md",
            "evaluation_logic.md",
            "data_sources.md",
            "routing_logic.md",
        ]

        for filename in required_files:
            filepath = DEFAULT_KNOWLEDGE_BASE_PATH / filename
            assert filepath.exists(), f"Missing knowledge base file: {filename}"

    def test_knowledge_base_files_have_content(self):
        from app.rag.ingest import DEFAULT_KNOWLEDGE_BASE_PATH

        ingester = DocumentIngester(knowledge_base_path=DEFAULT_KNOWLEDGE_BASE_PATH)
        docs = ingester.load_documents()

        assert len(docs) >= 6, "Expected at least 6 documents from knowledge base"

        total_content = sum(len(doc.content) for doc in docs)
        assert total_content > 5000, "Knowledge base should have substantial content"

        unique_sources = set(doc.source for doc in docs)
        assert len(unique_sources) >= 6, "Expected at least 6 unique source files"

        for source in unique_sources:
            source_docs = [d for d in docs if d.source == source]
            assert source_docs[0].title, f"Document {source} has no title"

    def test_retriever_finds_system_overview(self):
        retriever = create_retriever()

        results = retriever.retrieve("What is the Movie Night Assistant?")

        assert len(results) > 0
        sources = [r.metadata.get("source_file", "") for r in results]
        assert any("system_overview" in s for s in sources)

    def test_retriever_finds_limitations(self):
        retriever = create_retriever()

        results = retriever.retrieve("What are the known limitations?")

        assert len(results) > 0
        contents = " ".join(r.content.lower() for r in results)
        assert "limitation" in contents or "memory" in contents

    def test_retriever_finds_evaluation_logic(self):
        retriever = create_retriever()

        results = retriever.retrieve("How does evaluation work?")

        assert len(results) > 0
