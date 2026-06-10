"""ChromaDB-backed document store for the RAG pipeline.

Handles all ChromaDB interactions: client creation, collection management,
document indexing, and semantic similarity search.
"""

import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.rag.ingest import KnowledgeDocument
from app.schemas.domain import RetrievedContext

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "knowledge_base"


class ChromaDocumentStore:
    """Manages a ChromaDB collection for semantic document retrieval.

    Responsible for:
    - Creating or opening a ChromaDB collection (persistent or ephemeral).
    - Embedding and indexing KnowledgeDocument chunks on first use.
    - Performing similarity search and returning RetrievedContext results.
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        persist_directory: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialise the store configuration (no I/O performed yet).

        Args:
            embeddings: LangChain Embeddings used to embed documents and queries.
                If None, ChromaDB's built-in default embedding function is used.
            persist_directory: Directory for ChromaDB persistence.
                Pass None for an ephemeral in-memory collection.
            collection_name: Name of the ChromaDB collection.
        """
        self._embeddings = embeddings
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._chroma: Chroma | None = None

    def setup(self, documents: list[KnowledgeDocument]) -> None:
        """Open (or create) the collection and index documents if it is empty.

        Args:
            documents: All document chunks to index if the collection is new.
        """
        client = self._create_client()

        self._chroma = Chroma(
            client=client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

        existing_count = self._chroma._collection.count()
        logger.info(
            "Chroma collection '%s' contains %d existing documents",
            self._collection_name,
            existing_count,
        )

        if existing_count == 0:
            self._index_documents(documents)
        else:
            logger.info(
                "Skipping re-indexing — collection already populated with %d documents",
                existing_count,
            )

    def search(
        self, query: str, k: int, min_score: float
    ) -> list[RetrievedContext]:
        """Run a cosine-similarity search and return scored document chunks.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.
            min_score: Minimum relevance score (0–1) for a result to be included.

        Returns:
            Ordered list of RetrievedContext objects (highest relevance first).
        """
        if self._chroma is None:
            raise RuntimeError("ChromaDocumentStore.setup() must be called before search()")

        logger.info("Querying ChromaDB for top %d chunks: '%s'", k, query[:60])

        raw_results = self._chroma.similarity_search_with_relevance_scores(query, k=k)

        contexts = [
            self._to_retrieved_context(doc, score)
            for doc, score in raw_results
            if score >= min_score
        ]

        if contexts:
            logger.info(
                "Retrieved %d chunks (top relevance score: %.3f)",
                len(contexts),
                contexts[0].relevance_score or 0.0,
            )
        else:
            logger.info("No chunks met the minimum relevance threshold (%.2f)", min_score)

        return contexts

    def _create_client(self) -> chromadb.ClientAPI:
        """Return a persistent or ephemeral ChromaDB client."""
        if self._persist_directory:
            logger.info("Using persistent ChromaDB at '%s'", self._persist_directory)
            return chromadb.PersistentClient(path=self._persist_directory)

        logger.info("Using ephemeral (in-memory) ChromaDB collection")
        return chromadb.EphemeralClient()

    def _index_documents(self, documents: list[KnowledgeDocument]) -> None:
        """Embed and store document chunks in the Chroma collection."""
        logger.info(
            "Indexing %d chunks into ChromaDB collection '%s'",
            len(documents),
            self._collection_name,
        )

        texts = [doc.content for doc in documents]
        metadatas = [self._build_metadata(doc) for doc in documents]
        ids = [self._stable_document_id(doc) for doc in documents]

        self._chroma.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info("Successfully indexed %d chunks", len(documents))

    def _build_metadata(self, doc: KnowledgeDocument) -> dict:
        """Build the flat metadata dict persisted in ChromaDB for a chunk."""
        return {
            "title": doc.title,
            "source_file": doc.source,
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "total_chunks": doc.metadata.get("total_chunks", 1),
            "file_path": doc.metadata.get("file_path", ""),
        }

    def _stable_document_id(self, doc: KnowledgeDocument) -> str:
        """Return a stable unique ID for a chunk based on source file and index."""
        source_stem = doc.source.replace(".md", "")
        chunk_index = doc.metadata.get("chunk_index", 0)
        return f"{source_stem}_{chunk_index}"

    def _to_retrieved_context(self, doc, score: float) -> RetrievedContext:
        """Convert a Chroma search result into a RetrievedContext."""
        meta = doc.metadata
        return RetrievedContext(
            content=doc.page_content,
            source="rag",
            relevance_score=max(0.0, min(1.0, score)),
            metadata={
                "title": meta.get("title", ""),
                "source_file": meta.get("source_file", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "file_path": meta.get("file_path", ""),
            },
        )
