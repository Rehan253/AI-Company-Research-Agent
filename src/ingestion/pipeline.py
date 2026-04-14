"""
Ingestion Pipeline — orchestrates the full data ingestion flow.

Connects all ingestion components in sequence:
  Web Scraper → PDF Processor → Chunker → Embedder → Vector Store

Single entry point for ingesting a company:
  pipeline = IngestionPipeline()
  report = await pipeline.ingest_company("Danone")

The pipeline returns an IngestionReport with stats — useful for
the API response, logging, and demo purposes.

Design principle — separation of concerns:
  Each component (scraper, chunker, embedder, store) does ONE thing.
  The pipeline only coordinates — it doesn't contain business logic.
  This makes each component independently testable and replaceable.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.vector_store import VectorStore
from src.ingestion.web_scraper import CompanyWebScraper


@dataclass
class IngestionReport:
    """
    Summary of what was ingested.

    Returned after every pipeline run — useful for API responses
    and for showing progress in the UI demo.
    """
    company_name: str
    success: bool

    # Source statistics
    sources_scraped: int = 0
    pdfs_processed: int = 0
    total_documents: int = 0

    # Processing statistics
    total_chunks: int = 0
    total_characters: int = 0

    # Performance
    duration_seconds: float = 0.0

    # Any errors encountered (non-fatal — pipeline continues despite them)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"[{status}] {self.company_name} | "
            f"{self.total_documents} docs → {self.total_chunks} chunks | "
            f"{self.total_characters:,} chars | "
            f"{self.duration_seconds:.1f}s"
        )


class IngestionPipeline:
    """
    Orchestrates the full company data ingestion flow.

    Usage:
        pipeline = IngestionPipeline()

        # Ingest from web only
        report = await pipeline.ingest_company("Danone")

        # Ingest with official website
        report = await pipeline.ingest_company(
            "Danone",
            company_url="https://www.danone.com"
        )

        # Ingest with PDFs
        report = await pipeline.ingest_company(
            "Danone",
            pdf_paths=["danone_annual_report_2024.pdf"]
        )
    """

    def __init__(self):
        # Initialize all components once — they're reused across calls
        self.scraper = CompanyWebScraper()
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    async def ingest_company(
        self,
        company_name: str,
        company_url: str | None = None,
        pdf_paths: list[str | Path] | None = None,
        replace_existing: bool = False,
    ) -> IngestionReport:
        """
        Full ingestion pipeline for a company.

        Steps:
          1. Scrape Wikipedia + optional website
          2. Process any uploaded PDFs
          3. Chunk all collected text
          4. Generate embeddings
          5. Store in ChromaDB

        Args:
            company_name: Name of the company (e.g. "Danone", "LVMH")
            company_url: Optional official website URL
            pdf_paths: Optional list of PDF file paths to process
            replace_existing: If True, delete existing data before ingesting

        Returns:
            IngestionReport with stats and any errors
        """
        start_time = time.time()
        report = IngestionReport(company_name=company_name, success=False)

        logger.info(f"Starting ingestion for: {company_name}")

        # --- Optional: clear existing data ---
        if replace_existing:
            logger.info(f"Replacing existing data for: {company_name}")
            self.vector_store.delete_company(company_name)

        # --- Step 1: Collect documents ---
        all_documents: list[Document] = []

        # 1a. Web sources (Wikipedia + optional company site)
        try:
            web_docs = await self.scraper.scrape_company(company_name, company_url)
            all_documents.extend(web_docs)
            report.sources_scraped = len(web_docs)
            logger.info(f"Web scraping: {len(web_docs)} documents collected")
        except Exception as e:
            msg = f"Web scraping failed: {e}"
            logger.error(msg)
            report.errors.append(msg)

        # 1b. PDFs (if provided)
        if pdf_paths:
            for pdf_path in pdf_paths:
                try:
                    pdf_docs = self.pdf_processor.process_pdf(
                        pdf_path,
                        company_name=company_name,
                        doc_type="uploaded_document",
                    )
                    all_documents.extend(pdf_docs)
                    report.pdfs_processed += 1
                    logger.info(f"PDF processed: {Path(pdf_path).name} → {len(pdf_docs)} pages")
                except Exception as e:
                    msg = f"PDF processing failed for {pdf_path}: {e}"
                    logger.error(msg)
                    report.errors.append(msg)

        report.total_documents = len(all_documents)
        report.total_characters = sum(len(d.page_content) for d in all_documents)

        if not all_documents:
            logger.error(f"No documents collected for {company_name} — aborting")
            report.duration_seconds = time.time() - start_time
            return report

        # --- Step 2: Chunk ---
        try:
            chunks = self.chunker.chunk_documents(all_documents)
            report.total_chunks = len(chunks)
            logger.info(f"Chunking: {len(all_documents)} docs → {len(chunks)} chunks")
        except Exception as e:
            msg = f"Chunking failed: {e}"
            logger.error(msg)
            report.errors.append(msg)
            report.duration_seconds = time.time() - start_time
            return report

        # --- Step 3: Embed ---
        try:
            embeddings = self.embedder.embed_documents(chunks)
            logger.info(f"Embedding: {len(embeddings)} vectors generated")
        except Exception as e:
            msg = f"Embedding failed: {e}"
            logger.error(msg)
            report.errors.append(msg)
            report.duration_seconds = time.time() - start_time
            return report

        # --- Step 4: Store ---
        try:
            stored = self.vector_store.add_documents(company_name, chunks, embeddings)
            logger.info(f"Storage: {stored} chunks saved to ChromaDB")
        except Exception as e:
            msg = f"Storage failed: {e}"
            logger.error(msg)
            report.errors.append(msg)
            report.duration_seconds = time.time() - start_time
            return report

        # --- Done ---
        report.success = True
        report.duration_seconds = round(time.time() - start_time, 2)
        logger.success(f"Ingestion complete: {report}")
        return report

    def get_ingested_companies(self) -> list[str]:
        """List all companies currently in the vector store."""
        return self.vector_store.list_companies()

    def get_company_stats(self, company_name: str) -> dict:
        """Return stats for an already-ingested company."""
        return self.vector_store.get_company_stats(company_name)
