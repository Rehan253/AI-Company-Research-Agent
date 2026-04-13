"""
PDF Processor for extracting text from company documents.

Uses PyMuPDF (imported as 'fitz') to extract text page-by-page from:
  - Annual reports
  - Financial filings
  - Press releases
  - Investor presentations

Key concept — page-level metadata:
  Each page becomes its own Document with its page number stored in metadata.
  This flows through the entire pipeline (chunker → embedder → ChromaDB)
  so when the LLM cites a source it can say:
  "Source: Danone Annual Report 2024, page 12" — not just "some PDF".

Key concept — why page-by-page (not full document at once):
  A 200-page annual report is too large to embed as one chunk.
  Splitting by page gives natural semantic boundaries and precise citations.
  The chunker will further split long pages into smaller pieces later.
"""

from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF — the 'fitz' name is historical, it's PyMuPDF
from langchain_core.documents import Document
from loguru import logger


class PDFProcessor:
    """
    Extracts text from PDF files, one Document per page.

    Usage:
        processor = PDFProcessor()
        docs = processor.process_pdf("danone_annual_report_2024.pdf", company_name="Danone")
        # Returns list of Documents, one per page with metadata
    """

    # Minimum characters to consider a page worth keeping
    # Pages with less are usually blank, cover pages, or image-only
    MIN_PAGE_CHARS = 100

    def process_pdf(
        self,
        file_path: str | Path,
        company_name: str,
        doc_type: str = "annual_report",
    ) -> list[Document]:
        """
        Extract text from every page of a PDF.

        Args:
            file_path: Path to the PDF file
            company_name: Used for metadata tagging
            doc_type: Label for this document e.g. "annual_report", "press_release"

        Returns:
            List of Documents, one per meaningful page
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"PDF not found: {file_path}")
            return []

        if not file_path.suffix.lower() == ".pdf":
            logger.error(f"Not a PDF file: {file_path}")
            return []

        logger.info(f"Processing PDF: {file_path.name}")

        try:
            # fitz.open() loads the PDF into memory
            pdf = fitz.open(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF '{file_path.name}': {e}")
            return []

        # Extract PDF-level metadata (title, author, creation date)
        pdf_metadata = self._extract_pdf_metadata(pdf, file_path)
        logger.info(
            f"PDF: {pdf_metadata['title']} | "
            f"{pdf_metadata['page_count']} pages | "
            f"Author: {pdf_metadata['author']}"
        )

        documents = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]  # Zero-indexed internally
            display_page = page_num + 1  # Human-readable (1-indexed)

            text = self._extract_page_text(page)

            # Skip pages with too little content (blank, images, etc.)
            if len(text) < self.MIN_PAGE_CHARS:
                logger.debug(f"Skipping page {display_page} (only {len(text)} chars)")
                continue

            doc = Document(
                page_content=text,
                metadata={
                    # Source information for citations
                    "source": str(file_path),
                    "source_filename": file_path.name,
                    "page": display_page,
                    "total_pages": pdf_metadata["page_count"],

                    # Document classification
                    "company": company_name.lower(),
                    "doc_type": doc_type,
                    "title": pdf_metadata["title"],
                    "author": pdf_metadata["author"],

                    # Processing metadata
                    "processed_at": datetime.utcnow().isoformat(),
                },
            )
            documents.append(doc)

        pdf.close()

        skipped = pdf_metadata["page_count"] - len(documents)
        logger.info(
            f"Extracted {len(documents)} pages "
            f"({skipped} skipped — blank or image-only)"
        )
        return documents

    def process_pdfs_in_folder(
        self,
        folder_path: str | Path,
        company_name: str,
        doc_type: str = "report",
    ) -> list[Document]:
        """
        Process all PDFs in a folder at once.

        Useful when a company has multiple reports to ingest together.

        Args:
            folder_path: Directory containing PDF files
            company_name: Applied to all documents
            doc_type: Applied to all documents

        Returns:
            All extracted Documents from all PDFs combined
        """
        folder_path = Path(folder_path)
        pdf_files = list(folder_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in: {folder_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDFs in {folder_path}")
        all_documents = []

        for pdf_file in pdf_files:
            docs = self.process_pdf(pdf_file, company_name, doc_type)
            all_documents.extend(docs)

        logger.info(f"Total pages extracted from folder: {len(all_documents)}")
        return all_documents

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract clean text from a single PDF page.

        PyMuPDF gives us raw text with layout artifacts. We clean:
        - Excessive blank lines (from column layouts)
        - Hyphenated line breaks (words split across lines in PDFs)
        - Leading/trailing whitespace per line
        """
        # get_text("text") returns plain text preserving newlines
        raw_text = page.get_text("text")

        lines = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Rejoin hyphenated words split across lines
            # e.g. "sustain-\nability" → "sustainability"
            if lines and lines[-1].endswith("-"):
                lines[-1] = lines[-1][:-1] + line
            else:
                lines.append(line)

        return "\n".join(lines)

    def _extract_pdf_metadata(self, pdf: fitz.Document, file_path: Path) -> dict:
        """
        Extract document-level metadata from the PDF header.

        PDF files embed metadata: title, author, subject, creation date.
        Not all PDFs have this filled in — we fall back to filename.
        """
        meta = pdf.metadata or {}

        # Use filename as fallback title if PDF has no embedded title
        title = meta.get("title") or file_path.stem.replace("_", " ").replace("-", " ").title()

        return {
            "title": title,
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", ""),
            "creation_date": meta.get("creationDate", ""),
            "page_count": len(pdf),
        }
