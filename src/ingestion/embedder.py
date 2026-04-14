"""
Embedding Engine — converts text chunks into numerical vectors.

What embeddings do:
  Text → list of numbers (vector) that represents meaning.
  Similar meanings → similar vectors → close together in vector space.

  "Danone annual revenue" and "Danone sales figures" will have very
  similar vectors even though they share no words. This is semantic search.

Model: all-MiniLM-L6-v2
  - Free, runs locally (no API cost)
  - 384-dimensional output vectors
  - Fast: ~14,000 sentences/second on CPU
  - Good quality for information retrieval tasks

Interview talking point:
  "I chose local embeddings (sentence-transformers) to keep costs near zero
  during development. The code is designed so you can swap in OpenAI's
  text-embedding-3-small for production by changing one config value."
"""

import numpy as np
from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


class Embedder:
    """
    Generates embeddings for text chunks using sentence-transformers.

    Usage:
        embedder = Embedder()
        embeddings = embedder.embed_documents(chunks)
        # Returns numpy array of shape (num_chunks, 384)
    """

    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: Hugging Face model name (default from config)
        """
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")

        # SentenceTransformer downloads the model on first use (~90MB)
        # and caches it locally — subsequent loads are instant
        self._model = SentenceTransformer(self.model_name)

        # Store dimension for ChromaDB collection setup
        self.dimension = self._model.get_embedding_dimension()
        logger.info(f"Embedding model ready — {self.dimension}-dimensional vectors")

    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of Documents.

        Args:
            documents: Chunks from the TextChunker

        Returns:
            numpy array of shape (len(documents), self.dimension)
            Each row is the embedding vector for the corresponding document
        """
        if not documents:
            return np.array([])

        texts = [doc.page_content for doc in documents]
        logger.info(f"Embedding {len(texts)} chunks...")

        # encode() handles batching internally for efficiency
        # show_progress_bar gives visual feedback for large batches
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 20,
            normalize_embeddings=True,  # Normalize to unit length for cosine similarity
        )

        logger.info(f"Generated {len(embeddings)} embeddings ({self.dimension}d each)")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single search query.

        Used at query time to embed the user's question before
        searching ChromaDB for similar chunks.

        Args:
            query: The user's search question

        Returns:
            1D numpy array of shape (self.dimension,)
        """
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding
