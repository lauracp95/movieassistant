"""Unit tests for RAG retriever, ingest, and agent components."""

import hashlib
import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from app.agents.rag_agent import RAGAssistantAgent
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


class FakeEmbeddings:
    """Deterministic hash-based embeddings for unit tests (no Azure calls)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in digest]


def unique_collection() -> str:
    """Return a unique ChromaDB collection name to isolate each test."""
    return f"test_{uuid.uuid4().hex}"


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
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )
            results = retriever.retrieve("any query")
            assert results == []

    def test_retrieve_returns_results_for_valid_query(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "movies.md").write_text(
                "# Movies\n\nThis document covers movie recommendations."
            )
            (Path(tmpdir) / "system.md").write_text(
                "# System\n\nThis document explains system architecture."
            )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                top_k=2,
                min_score=0.0,
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("movie recommendations")

            assert len(results) >= 1
            assert len(results) <= 2

    def test_retrieve_respects_top_k(self):
        with TemporaryDirectory() as tmpdir:
            for i in range(5):
                (Path(tmpdir) / f"doc{i}.md").write_text(
                    f"# Document {i}\n\nThis is document number {i} about topics."
                )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                top_k=2,
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("document topics")

            assert len(results) <= 2

    def test_retrieve_respects_top_k_override(self):
        with TemporaryDirectory() as tmpdir:
            for i in range(5):
                (Path(tmpdir) / f"doc{i}.md").write_text(
                    f"# Document {i}\n\nContent for document {i}."
                )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                top_k=5,
                min_score=0.0,
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("document content", top_k=1)

            assert len(results) <= 1

    def test_retrieve_returns_retrieved_context_objects(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text(
                "# Test\n\nContent for testing retrieval."
            )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                min_score=0.0,
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("testing retrieval")

            assert len(results) >= 1
            for r in results:
                assert isinstance(r, RetrievedContext)
                assert r.source == "rag"
                assert r.relevance_score is not None
                assert 0.0 <= r.relevance_score <= 1.0
                assert "title" in r.metadata

    def test_retrieve_empty_string_returns_empty(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("")
            assert results == []

    def test_retrieve_whitespace_only_query_returns_empty(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("   ")
            assert results == []

    def test_retrieve_preserves_metadata(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "my_doc.md").write_text(
                "# My Great Document\n\nSome content here."
            )

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                min_score=0.0,
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve("content")

            assert len(results) >= 1
            meta = results[0].metadata
            assert meta["title"] == "My Great Document"
            assert meta["source_file"] == "my_doc.md"
            assert "chunk_index" in meta
            assert "total_chunks" in meta
            assert "file_path" in meta

    def test_retrieve_all_returns_all_documents(self):
        with TemporaryDirectory() as tmpdir:
            for i in range(3):
                (Path(tmpdir) / f"doc{i}.md").write_text(f"# Doc {i}\n\nContent {i}.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve_all()

            assert len(results) == 3
            for r in results:
                assert r.relevance_score == 1.0
                assert r.source == "rag"

    def test_retrieve_all_preserves_metadata(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "sample.md").write_text("# Sample\n\nA short document.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )
            retriever.initialize()

            results = retriever.retrieve_all()

            assert len(results) == 1
            meta = results[0].metadata
            assert meta["title"] == "Sample"
            assert meta["source_file"] == "sample.md"

    def test_auto_initialize_on_first_retrieve(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )

            assert not retriever._initialized
            retriever.retrieve("anything")
            assert retriever._initialized

    def test_initialize_is_idempotent(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )

            retriever.initialize()
            retriever.initialize()

            assert retriever._initialized
            assert len(retriever._documents) >= 1

    def test_empty_knowledge_base_does_not_crash(self):
        with TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = DocumentRetriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )
            retriever.initialize()

            assert retriever._initialized
            assert retriever._store is None
            assert retriever.retrieve("any query") == []
            assert retriever.retrieve_all() == []

    def test_create_retriever_factory_function(self):
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test\n\nContent.")

            ingester = DocumentIngester(knowledge_base_path=Path(tmpdir))
            retriever = create_retriever(
                ingester=ingester,
                embeddings=FakeEmbeddings(),
                collection_name=unique_collection(),
            )

            assert retriever._initialized
            assert len(retriever._documents) >= 1


class TestRAGAssistantAgent:
    def test_answer_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="This is the answer based on documentation."
        )

        agent = RAGAssistantAgent(mock_llm)
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

        agent = RAGAssistantAgent(mock_llm)
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

        agent = RAGAssistantAgent(mock_llm)
        answer = agent.answer("query", [])

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        user_message = call_args[1].content

        assert "No relevant documentation found" in user_message


_AZURE_EMBEDDINGS_CONFIGURED = bool(os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"))


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

    def test_knowledge_base_retriever_initializes_with_fake_embeddings(self):
        from app.rag.ingest import DEFAULT_KNOWLEDGE_BASE_PATH

        ingester = DocumentIngester(knowledge_base_path=DEFAULT_KNOWLEDGE_BASE_PATH)
        retriever = create_retriever(
            ingester=ingester,
            embeddings=FakeEmbeddings(),
            min_score=0.0,
            collection_name=unique_collection(),
        )

        assert retriever._initialized
        assert len(retriever._documents) >= 6

    def test_retrieve_all_returns_all_kb_documents(self):
        from app.rag.ingest import DEFAULT_KNOWLEDGE_BASE_PATH

        ingester = DocumentIngester(knowledge_base_path=DEFAULT_KNOWLEDGE_BASE_PATH)
        retriever = create_retriever(
            ingester=ingester,
            embeddings=FakeEmbeddings(),
            collection_name=unique_collection(),
        )

        results = retriever.retrieve_all()

        assert len(results) >= 6
        for r in results:
            assert r.relevance_score == 1.0
            assert r.metadata.get("title")
            assert r.metadata.get("source_file")

    @pytest.mark.skipif(
        not _AZURE_EMBEDDINGS_CONFIGURED,
        reason="Requires AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT and Azure credentials",
    )
    def test_retriever_finds_system_overview(self):
        retriever = create_retriever()

        results = retriever.retrieve("What is the Movie Night Assistant?")

        assert len(results) > 0
        sources = [r.metadata.get("source_file", "") for r in results]
        assert any("system_overview" in s for s in sources)

    @pytest.mark.skipif(
        not _AZURE_EMBEDDINGS_CONFIGURED,
        reason="Requires AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT and Azure credentials",
    )
    def test_retriever_finds_limitations(self):
        retriever = create_retriever()

        results = retriever.retrieve("What are the known limitations?")

        assert len(results) > 0
        contents = " ".join(r.content.lower() for r in results)
        assert "limitation" in contents or "memory" in contents

    @pytest.mark.skipif(
        not _AZURE_EMBEDDINGS_CONFIGURED,
        reason="Requires AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT and Azure credentials",
    )
    def test_retriever_finds_evaluation_logic(self):
        retriever = create_retriever()

        results = retriever.retrieve("How does evaluation work?")

        assert len(results) > 0
