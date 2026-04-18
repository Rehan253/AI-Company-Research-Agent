"""
RAG Chain — combines Retriever + Generator into one callable.

This is the complete question-answering pipeline:
  User question
      ↓
  Retriever: find relevant chunks from ChromaDB (with MMR)
      ↓
  Generator: send chunks + question to GPT-4o-mini
      ↓
  GeneratedAnswer: text + citations + confidence

Usage:
    chain = RAGChain()
    answer = chain.ask("What is Danone's revenue?", company="Danone")
    print(answer.format_with_sources())
"""

from loguru import logger

from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import VectorStore
from src.rag.generator import GeneratedAnswer, Generator
from src.rag.retriever import Retriever


class RAGChain:
    """
    End-to-end RAG pipeline: retrieve → generate → cite.

    Holds shared instances of Embedder and VectorStore so the
    embedding model is only loaded once across multiple questions.
    """

    def __init__(self, n_results: int = 5):
        """
        Args:
            n_results: How many chunks to retrieve per question (top-k)
        """
        self.n_results = n_results

        # Shared across retriever — embedding model loads once (~2s)
        embedder = Embedder()
        vector_store = VectorStore()

        self.retriever = Retriever(vector_store=vector_store, embedder=embedder)
        self.generator = Generator()

    def ask(self, query: str, company_name: str) -> GeneratedAnswer:
        """
        Answer a question about a company using RAG.

        Args:
            query: Natural language question
            company_name: Company whose data to search

        Returns:
            GeneratedAnswer with text, citations, and confidence
        """
        logger.info(f"RAG query: '{query}' | company={company_name}")

        # Step 1: Retrieve relevant chunks (MMR for diversity)
        context_docs, context_string = self.retriever.retrieve_with_context(
            query=query,
            company_name=company_name,
            n_results=self.n_results,
        )

        if not context_docs:
            logger.warning(f"No context found for '{company_name}' — not yet ingested?")

        # Step 2: Generate cited answer
        answer = self.generator.generate(
            query=query,
            context_docs=context_docs,
            context_string=context_string,
        )

        logger.info(
            f"Answer generated | confidence={answer.confidence} | "
            f"citations={len(answer.citations)} | "
            f"had_context={answer.had_enough_context}"
        )
        return answer
