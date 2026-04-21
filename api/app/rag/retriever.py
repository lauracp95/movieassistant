"""Document retrieval for the RAG pipeline.

This module provides semantic search over the knowledge base documents
using simple TF-IDF-based similarity scoring.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.rag.ingest import DocumentIngester, KnowledgeDocument
from app.schemas.domain import RetrievedContext

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
MIN_RELEVANCE_SCORE = 0.1


@dataclass
class ScoredDocument:
    """A document with its relevance score."""

    document: KnowledgeDocument
    score: float


class DocumentRetriever:
    """Retrieves relevant documents from the knowledge base.

    Uses TF-IDF-based similarity scoring for semantic search.
    This is a simple implementation that works well for small
    document collections without requiring external dependencies.
    """

    def __init__(
        self,
        ingester: DocumentIngester | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> None:
        """Initialize the retriever.

        Args:
            ingester: DocumentIngester instance with loaded documents.
                If None, a new ingester is created and documents are loaded.
            top_k: Number of top documents to retrieve.
            min_score: Minimum relevance score threshold.
        """
        self._ingester = ingester or DocumentIngester()
        self._top_k = top_k
        self._min_score = min_score
        self._documents: list[KnowledgeDocument] = []
        self._doc_vectors: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Load documents and build the search index.

        Call this method before retrieving documents, or it will
        be called automatically on first retrieval.
        """
        if self._initialized:
            return

        self._documents = self._ingester.load_documents()

        if not self._documents:
            logger.warning("No documents loaded for retrieval")
            self._initialized = True
            return

        self._build_index()
        self._initialized = True
        logger.info(f"Retriever initialized with {len(self._documents)} documents")

    def _build_index(self) -> None:
        """Build TF-IDF index for all documents."""
        self._doc_vectors = [
            self._tokenize(doc.content) for doc in self._documents
        ]

        doc_count = len(self._documents)
        term_doc_counts: Counter = Counter()

        for vec in self._doc_vectors:
            for term in vec.keys():
                term_doc_counts[term] += 1

        self._idf = {
            term: math.log(doc_count / (1 + count))
            for term, count in term_doc_counts.items()
        }

    def _tokenize(self, text: str) -> Counter:
        """Tokenize text into a term frequency counter.

        Args:
            text: The text to tokenize.

        Returns:
            Counter of term frequencies.
        """
        text = text.lower()
        words = re.findall(r"\b[a-z]+\b", text)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "this", "that", "these", "those",
            "it", "its", "they", "them", "their", "what", "which", "who",
        }
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        return Counter(filtered)

    def _compute_similarity(
        self, query_vec: Counter, doc_vec: Counter
    ) -> float:
        """Compute TF-IDF cosine similarity between query and document.

        Args:
            query_vec: Query term frequencies.
            doc_vec: Document term frequencies.

        Returns:
            Similarity score between 0 and 1.
        """
        common_terms = set(query_vec.keys()) & set(doc_vec.keys())

        if not common_terms:
            return 0.0

        query_tfidf = {
            term: freq * self._idf.get(term, 0.0)
            for term, freq in query_vec.items()
        }
        doc_tfidf = {
            term: freq * self._idf.get(term, 0.0)
            for term, freq in doc_vec.items()
        }

        dot_product = sum(
            query_tfidf.get(term, 0.0) * doc_tfidf.get(term, 0.0)
            for term in common_terms
        )

        query_norm = math.sqrt(sum(v ** 2 for v in query_tfidf.values()))
        doc_norm = math.sqrt(sum(v ** 2 for v in doc_tfidf.values()))

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of results to return (overrides default).

        Returns:
            List of RetrievedContext objects with relevant documents.
        """
        if not self._initialized:
            self.initialize()

        if not self._documents:
            logger.info("No documents available for retrieval")
            return []

        k = top_k if top_k is not None else self._top_k
        query_vec = self._tokenize(query)

        if not query_vec:
            logger.info(f"Query produced no tokens: {query}")
            return []

        scored_docs = []
        for i, doc_vec in enumerate(self._doc_vectors):
            score = self._compute_similarity(query_vec, doc_vec)
            if score >= self._min_score:
                scored_docs.append(ScoredDocument(
                    document=self._documents[i],
                    score=score,
                ))

        scored_docs.sort(key=lambda x: x.score, reverse=True)
        top_docs = scored_docs[:k]

        logger.info(
            f"Retrieved {len(top_docs)} documents for query: {query[:50]}..."
        )

        return [
            RetrievedContext(
                content=doc.document.content,
                source="rag",
                relevance_score=doc.score,
                metadata={
                    "title": doc.document.title,
                    "source_file": doc.document.source,
                    **doc.document.metadata,
                },
            )
            for doc in top_docs
        ]

    def retrieve_all(self) -> list[RetrievedContext]:
        """Retrieve all documents without filtering.

        Useful for debugging or when all context is needed.

        Returns:
            List of all documents as RetrievedContext objects.
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
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> DocumentRetriever:
    """Factory function to create and initialize a retriever.

    Args:
        ingester: Optional DocumentIngester instance.
        top_k: Number of top documents to retrieve.
        min_score: Minimum relevance score threshold.

    Returns:
        Initialized DocumentRetriever instance.
    """
    retriever = DocumentRetriever(
        ingester=ingester,
        top_k=top_k,
        min_score=min_score,
    )
    retriever.initialize()
    return retriever
