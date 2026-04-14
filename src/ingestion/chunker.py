"""
Text Chunker — splits Documents into smaller pieces for embedding.

Why chunking matters for RAG quality:
  - Too large: retrieval is imprecise, LLM context gets flooded
  - Too small: chunks lose context, answers become fragmented
  - Just right (500-1000 tokens): precise retrieval + enough context per chunk

We use RecursiveCharacterTextSplitter which splits on natural boundaries:
  1. Paragraphs (\n\n)  ← preferred split point
  2. Lines (\n)
  3. Sentences (". ")
  4. Words (" ")
  5. Characters         ← last resort only

This preserves meaning better than splitting at a fixed character count.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings


class TextChunker:
    """
    Splits Documents into smaller chunks ready for embedding.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk_documents(documents)
        # Each chunk is a Document with enriched metadata
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Args:
            chunk_size: Max characters per chunk (default from config: 800)
            chunk_overlap: Characters shared between adjacent chunks (default: 100)

        Note: LangChain measures chunk size in characters by default.
        800 characters ≈ 150-200 tokens ≈ 1-2 paragraphs of normal text.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # RecursiveCharacterTextSplitter is LangChain's recommended splitter
        # It respects natural text boundaries (paragraphs → sentences → words)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # These separators are tried in order — paragraph breaks first
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,  # Count characters (not tokens)
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of Documents into chunks.

        Each chunk:
        - Inherits all metadata from its parent document
        - Gets additional metadata: chunk_index, total_chunks, source_document
        - Is filtered if too short to be useful

        Args:
            documents: Raw Documents from web scraper or PDF processor

        Returns:
            List of chunk Documents, ready for embedding
        """
        if not documents:
            logger.warning("No documents provided to chunker")
            return []

        all_chunks = []

        for doc in documents:
            chunks = self._chunk_single_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _chunk_single_document(self, document: Document) -> list[Document]:
        """
        Split one Document into chunks with enriched metadata.

        Args:
            document: A single Document (e.g. one Wikipedia article or one PDF page)

        Returns:
            List of chunk Documents with index metadata added
        """
        # Split the text — returns list of strings
        text_chunks = self.splitter.split_text(document.page_content)

        # Filter out chunks that are too short to be useful
        # (e.g. page headers, "Table of Contents", single words)
        text_chunks = [t for t in text_chunks if len(t.strip()) >= 50]

        if not text_chunks:
            logger.debug(f"No valid chunks from: {document.metadata.get('source', 'unknown')}")
            return []

        # Build a Document for each chunk, carrying over parent metadata
        chunks = []
        for i, text in enumerate(text_chunks):
            # Start with a copy of the parent's metadata
            chunk_metadata = dict(document.metadata)

            # Add chunk-specific metadata for traceability
            chunk_metadata.update({
                "chunk_index": i,           # Position within this document
                "total_chunks": len(text_chunks),  # How many chunks this doc produced
                "chunk_size": len(text),    # Actual size of this chunk
            })

            chunks.append(Document(
                page_content=text,
                metadata=chunk_metadata,
            ))

        return chunks
