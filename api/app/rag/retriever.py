"""Document retrieval for the RAG pipeline.

Orchestrates document loading (via DocumentIngester) and semantic search
(via ChromaDocumentStore).  All ChromaDB details are encapsulated in
chroma_store.py; this module has no direct ChromaDB dependency.
"""

import logging

from langchain_core.embeddings import Embeddings

from app.rag.chroma_store import ChromaDocumentStore, DEFAULT_COLLECTION_NAME
from app.rag.ingest import DocumentIngester, KnowledgeDocument
from app.schemas.domain import RetrievedContext

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
MIN_RELEVANCE_SCORE = 0.0


class DocumentRetriever:
    """Retrieves relevant documents from the knowledge base using semantic search.

    Loads document chunks from DocumentIngester, delegates indexing and search
    to ChromaDocumentStore, and exposes a simple retrieve / retrieve_all API.
    """

    def __init__(
        self,
        ingester: DocumentIngester | None = None,
        embeddings: Embeddings | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_RELEVANCE_SCORE,
        persist_directory: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialise the retriever.

        Args:
            ingester: DocumentIngester that provides KnowledgeDocument chunks.
                A default instance is created if not supplied.
            embeddings: LangChain Embeddings used to embed queries and documents.
                If None, ChromaDB's built-in default embedding function is used.
            top_k: Default number of chunks to return per query.
            min_score: Minimum relevance score (0–1) for a chunk to be returned.
            persist_directory: Directory for ChromaDB persistence.
                Use None for an ephemeral in-memory collection.
            collection_name: Name of the ChromaDB collection.
        """
        self._ingester = ingester or DocumentIngester()
        self._top_k = top_k
        self._min_score = min_score
        self._store_config = dict(
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self._documents: list[KnowledgeDocument] = []
        self._store: ChromaDocumentStore | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load documents and set up the vector store.

        Safe to call multiple times; subsequent calls are no-ops.
        Called automatically on the first call to `retrieve` or `retrieve_all`.
        """
        if self._initialized:
            return

        logger.info("Initializing DocumentRetriever")

        self._documents = self._ingester.load_documents()

        if not self._documents:
            logger.warning("Knowledge base is empty — no documents available for retrieval")
            self._initialized = True
            return

        self._store = ChromaDocumentStore(**self._store_config)
        self._store.setup(self._documents)

        self._initialized = True
        logger.info(
            "DocumentRetriever ready: %d chunks available", len(self._documents)
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedContext]:
        """Retrieve the most relevant document chunks for a query.

        Args:
            query: The natural-language search query.
            top_k: Number of results to return; overrides the instance default.

        Returns:
            Ordered list of RetrievedContext objects (highest relevance first).
        """
        if not self._initialized:
            self.initialize()

        if not query.strip():
            logger.info("Empty query received — returning no results")
            return []

        if not self._documents or self._store is None:
            logger.info("No documents available for retrieval")
            return []

        k = top_k if top_k is not None else self._top_k
        return self._store.search(query, k=k, min_score=self._min_score)

    def retrieve_all(self) -> list[RetrievedContext]:
        """Return every indexed document chunk without filtering.

        Useful for debugging or when the full knowledge base is needed as context.

        Returns:
            All chunks as RetrievedContext objects with relevance_score = 1.0.
        """
        if not self._initialized:
            self.initialize()

        return [
            RetrievedContext(
                content=doc.content,
                source="rag",
                relevance_score=1.0,
                metadata={
                    "title": doc.title,
                    "source_file": doc.source,
                    **doc.metadata,
                },
            )
            for doc in self._documents
        ]


def create_retriever(
    ingester: DocumentIngester | None = None,
    embeddings: Embeddings | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> DocumentRetriever:
    """Create and initialise a DocumentRetriever.

    Args:
        ingester: Optional DocumentIngester; a default one is created if None.
        embeddings: LangChain Embeddings for semantic search.
        top_k: Number of top chunks to return per query.
        min_score: Minimum relevance score threshold (0–1).
        persist_directory: Path for ChromaDB persistence; None = in-memory.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Fully initialised DocumentRetriever ready for retrieval.
    """
    retriever = DocumentRetriever(
        ingester=ingester,
        embeddings=embeddings,
        top_k=top_k,
        min_score=min_score,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    retriever.initialize()
    return retriever
