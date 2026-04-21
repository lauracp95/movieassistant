"""RAG (Retrieval-Augmented Generation) package for the Movie Night Assistant.

This package provides document retrieval capabilities for answering
questions about the system using internal documentation.
"""

from app.rag.ingest import DocumentIngester, KnowledgeDocument
from app.rag.retriever import DocumentRetriever

__all__ = [
    "DocumentIngester",
    "DocumentRetriever",
    "KnowledgeDocument",
]
