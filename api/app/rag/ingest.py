"""Document ingestion for the RAG knowledge base.

This module handles loading and chunking markdown documents from the
knowledge base directory for retrieval-augmented generation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class KnowledgeDocument:
    """A document from the knowledge base.

    Attributes:
        content: The text content of the document or chunk.
        source: The source file name.
        title: The document title (from first heading or filename).
        metadata: Additional metadata about the document.
    """

    content: str
    source: str
    title: str
    metadata: dict = field(default_factory=dict)


class DocumentIngester:
    """Ingests markdown documents from the knowledge base.

    Handles loading, parsing, and chunking of markdown files for
    use in retrieval-augmented generation.
    """

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize the document ingester.

        Args:
            knowledge_base_path: Path to the knowledge base directory.
                Defaults to the knowledge_base folder in this package.
            chunk_size: Target size for text chunks in characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self._knowledge_base_path = knowledge_base_path or DEFAULT_KNOWLEDGE_BASE_PATH
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._documents: list[KnowledgeDocument] = []

    @property
    def documents(self) -> list[KnowledgeDocument]:
        """Get all loaded documents."""
        return self._documents

    def load_documents(self) -> list[KnowledgeDocument]:
        """Load all markdown documents from the knowledge base.

        Returns:
            List of KnowledgeDocument objects.
        """
        self._documents = []

        if not self._knowledge_base_path.exists():
            logger.warning(
                f"Knowledge base path does not exist: {self._knowledge_base_path}"
            )
            return self._documents

        md_files = list(self._knowledge_base_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files in knowledge base")

        for md_file in md_files:
            try:
                docs = self._load_file(md_file)
                self._documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} chunks from {md_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {md_file.name}: {e}")

        logger.info(f"Total documents loaded: {len(self._documents)}")
        return self._documents

    def _load_file(self, file_path: Path) -> list[KnowledgeDocument]:
        """Load and chunk a single markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            List of KnowledgeDocument objects (one per chunk).
        """
        content = file_path.read_text(encoding="utf-8")
        title = self._extract_title(content, file_path.stem)
        source = file_path.name

        chunks = self._chunk_text(content)

        return [
            KnowledgeDocument(
                content=chunk,
                source=source,
                title=title,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_path": str(file_path),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract the title from markdown content.

        Looks for the first H1 heading (# Title) in the content.

        Args:
            content: The markdown content.
            fallback: Fallback title if no heading found.

        Returns:
            The extracted title or fallback.
        """
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return fallback.replace("_", " ").title()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Attempts to split on paragraph boundaries when possible,
        falling back to character-based splitting.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            if not para.strip():
                continue

            if len(current_chunk) + len(para) + 2 <= self._chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(para) > self._chunk_size:
                    para_chunks = self._chunk_long_paragraph(para)
                    chunks.extend(para_chunks[:-1])
                    current_chunk = para_chunks[-1] if para_chunks else ""
                else:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + para if overlap_text else para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _chunk_long_paragraph(self, para: str) -> list[str]:
        """Chunk a paragraph that exceeds chunk_size.

        Args:
            para: The long paragraph to chunk.

        Returns:
            List of chunks from the paragraph.
        """
        chunks = []
        sentences = para.replace(". ", ".\n").split("\n")

        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self._chunk_size:
                current = current + " " + sentence if current else sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks if chunks else [para]

    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of text.

        Args:
            text: The text to get overlap from.

        Returns:
            The overlap text, or empty string if text is too short.
        """
        if len(text) <= self._chunk_overlap:
            return ""

        overlap = text[-self._chunk_overlap:]
        space_idx = overlap.find(" ")
        if space_idx > 0:
            return overlap[space_idx + 1:]
        return overlap
