"""
Vector Store — stores and retrieves embedded chunks using ChromaDB.

What ChromaDB does:
  Stores (text, embedding_vector, metadata) tuples.
  Given a query vector, finds the N most similar stored vectors
  using approximate nearest neighbor search (HNSW algorithm).

  This is how "semantic search" works — you ask a question,
  we embed it, find the closest chunk vectors, return their text.

Design decision — one collection per company:
  We create a separate ChromaDB collection for each company ingested.
  This means queries are automatically scoped to one company's data.
  "Search Danone's collection" is faster and more precise than
  "search everything and filter by metadata afterwards."

Interview talking point:
  "I chose ChromaDB for development because it's free, local, and needs
  zero infrastructure. The abstraction I built makes it easy to swap in
  Qdrant or Pinecone for production by replacing this file."
"""

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from loguru import logger

from src.config import settings


class VectorStore:
    """
    Manages ChromaDB collections for storing and retrieving company chunks.

    Usage:
        store = VectorStore()
        store.add_documents("danone", chunks, embeddings)
        results = store.query("danone", query_embedding, n_results=5)
    """

    def __init__(self, persist_dir: str | None = None):
        """
        Args:
            persist_dir: Directory to save ChromaDB data (default from config)
                         Data is saved to disk so it survives restarts.
        """
        self.persist_dir = persist_dir or settings.chroma_persist_dir

        # PersistentClient saves data to disk automatically after every operation
        # anonymized_telemetry=False: don't send usage stats to ChromaDB
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialized at: {self.persist_dir}")

    def add_documents(
        self,
        company_name: str,
        documents: list[Document],
        embeddings: list,
    ) -> int:
        """
        Store chunks and their embeddings in ChromaDB.

        Args:
            company_name: Used as the collection name (e.g. "danone")
            documents: Chunks from TextChunker (text + metadata)
            embeddings: Numpy array from Embedder, one row per document

        Returns:
            Number of chunks successfully stored
        """
        if not documents:
            logger.warning("No documents to store")
            return 0

        collection = self._get_or_create_collection(company_name)

        # ChromaDB requires string IDs — we generate unique ones
        # Using existing count ensures no ID collisions on re-ingestion
        existing_count = collection.count()
        ids = [f"chunk_{existing_count + i}" for i in range(len(documents))]

        # ChromaDB stores three things together per entry:
        #   - documents: the raw text (for retrieval)
        #   - embeddings: the vector (for similarity search)
        #   - metadatas: source info (for citations)
        collection.add(
            ids=ids,
            documents=[doc.page_content for doc in documents],
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=[doc.metadata for doc in documents],
        )

        logger.info(f"Stored {len(documents)} chunks in collection '{company_name}'")
        return len(documents)

    def query(
        self,
        company_name: str,
        query_embedding: list,
        n_results: int = 5,
    ) -> list[Document]:
        """
        Find the most semantically similar chunks to a query.

        This is the core of RAG retrieval — given a question embedding,
        find the chunks whose embeddings are most similar (nearest neighbors).

        Args:
            company_name: Which company's collection to search
            query_embedding: Embedded query from Embedder.embed_query()
            n_results: How many chunks to return (top-k)

        Returns:
            List of Documents ordered by relevance (most relevant first)
        """
        collection = self._get_collection(company_name)
        if collection is None:
            logger.warning(f"No collection found for company: '{company_name}'")
            return []

        # Cap results to what's actually stored
        n_results = min(n_results, collection.count())
        if n_results == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested response format into clean Documents
        documents = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # distance is cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity score (1 = identical, -1 = opposite)
            similarity = 1 - distance

            meta = dict(metadata)
            meta["relevance_score"] = round(similarity, 4)

            documents.append(Document(page_content=text, metadata=meta))

        return documents

    def delete_company(self, company_name: str) -> bool:
        """
        Remove all data for a company (delete its collection).

        Used when re-ingesting or cleaning up old data.
        """
        collection_name = self._collection_name(company_name)
        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection: '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete '{collection_name}': {e}")
            return False

    def list_companies(self) -> list[str]:
        """List all companies that have been ingested."""
        collections = self._client.list_collections()
        return [col.name for col in collections]

    def get_company_stats(self, company_name: str) -> dict:
        """Return chunk count and other stats for a company."""
        collection = self._get_collection(company_name)
        if collection is None:
            return {"exists": False, "chunk_count": 0}
        return {
            "exists": True,
            "chunk_count": collection.count(),
            "collection_name": self._collection_name(company_name),
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _collection_name(self, company_name: str) -> str:
        """Normalize company name to a valid ChromaDB collection name."""
        # ChromaDB collection names must be lowercase, no spaces
        return company_name.lower().replace(" ", "_").replace("-", "_")

    def _get_or_create_collection(self, company_name: str):
        """Get existing collection or create a new one."""
        name = self._collection_name(company_name)
        # cosine distance is best for normalized text embeddings
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_collection(self, company_name: str):
        """Get an existing collection, or None if it doesn't exist."""
        name = self._collection_name(company_name)
        try:
            return self._client.get_collection(name)
        except Exception:
            return None
