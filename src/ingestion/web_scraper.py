"""
Web Scraper for company data collection.

Uses httpx (async HTTP client) + BeautifulSoup (HTML parser) to extract
clean text from company websites, Wikipedia pages, and news articles.

Key concept — async:
  We use async/await so multiple URLs can be fetched concurrently.
  Instead of waiting for each request sequentially, we fire them all
  and collect results — much faster for scraping multiple pages.

Key concept — LangChain Document:
  Everything returns Document objects: {page_content: str, metadata: dict}
  This is the universal data format used by chunkers, vector stores,
  and retrievers throughout the project.
"""

import asyncio
from datetime import datetime
from urllib.parse import quote

import httpx
import wikipediaapi
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from loguru import logger


class CompanyWebScraper:
    """
    Scrapes company information from the web.

    Usage:
        scraper = CompanyWebScraper()
        docs = await scraper.scrape_company("Danone")
        # Returns list of Document objects ready for chunking
    """

    # Wikipedia API requires a descriptive User-Agent identifying your app
    # Generic browser agents are blocked by Wikipedia's API
    HEADERS = {
        "User-Agent": "AICompanyResearchAgent/1.0 (portfolio project; educational use) httpx/0.28",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Separate headers for regular website scraping (browser-like)
    SCRAPE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # HTML tags that contain the actual content (not navigation/ads)
    CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "li", "article", "section"]

    # Tags to remove entirely before extracting text
    NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]

    def __init__(self, timeout: int = 15, max_retries: int = 2):
        """
        Args:
            timeout: Seconds to wait for a response before giving up
            max_retries: How many times to retry on failure
        """
        self.timeout = timeout
        self.max_retries = max_retries

    async def scrape_url(self, url: str, company_name: str, doc_type: str = "web") -> Document | None:
        """
        Fetch a single URL and return a Document with clean text.

        Args:
            url: The webpage to scrape
            company_name: Used to tag the document metadata
            doc_type: Type label e.g. "wikipedia", "website", "news"

        Returns:
            Document object or None if scraping failed
        """
        # httpx.AsyncClient is like requests.Session but async
        # We create it fresh per call to avoid connection state issues
        async with httpx.AsyncClient(
            headers=self.SCRAPE_HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Scraping: {url} (attempt {attempt + 1})")
                    response = await client.get(url)
                    response.raise_for_status()  # Raises error if 4xx/5xx

                    text = self._extract_text(response.text)

                    if len(text) < 100:
                        logger.warning(f"Very little text extracted from {url}")
                        return None

                    return Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "company": company_name.lower(),
                            "doc_type": doc_type,
                            "scraped_at": datetime.utcnow().isoformat(),
                            "title": self._extract_title(response.text),
                        },
                    )

                except httpx.HTTPStatusError as e:
                    logger.warning(f"HTTP {e.response.status_code} for {url}")
                    return None
                except httpx.TimeoutException:
                    logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2)  # Wait before retry
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    return None

        return None

    async def scrape_wikipedia(self, company_name: str) -> Document | None:
        """
        Fetch the full Wikipedia article for a company using wikipedia-api.

        We use the wikipedia-api library instead of scraping HTML directly
        because it handles authentication, rate limits, and encoding correctly.
        It runs in a thread pool so it doesn't block our async event loop.

        Args:
            company_name: e.g. "Danone", "LVMH", "TotalEnergies"
        """
        logger.info(f"Fetching Wikipedia article for: {company_name}")

        # Wikipedia API is synchronous — run it in a thread pool
        # so it doesn't block our async event loop
        # asyncio.to_thread = "run this blocking function without freezing everything else"
        def _fetch():
            wiki = wikipediaapi.Wikipedia(
                user_agent="AICompanyResearchAgent/1.0 (educational portfolio project)",
                language="en",
            )
            page = wiki.page(company_name)
            if not page.exists():
                return None
            return page

        try:
            page = await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.error(f"Wikipedia fetch failed for '{company_name}': {e}")
            return None

        if page is None:
            logger.warning(f"Wikipedia: no page found for '{company_name}'")
            return None

        # Use full text for richer context (not just the intro summary)
        text = page.text
        if len(text) < 50:
            logger.warning(f"Wikipedia: empty page for '{company_name}'")
            return None

        page_url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
        logger.info(f"Wikipedia: {len(text)} characters for '{company_name}'")

        return Document(
            page_content=text,
            metadata={
                "source": page_url,
                "company": company_name.lower(),
                "doc_type": "wikipedia",
                "scraped_at": datetime.utcnow().isoformat(),
                "title": page.title,
            },
        )

    async def scrape_company_site(self, company_name: str, company_url: str) -> list[Document]:
        """
        Scrape the main pages of a company's official website.

        We scrape the homepage and /about page as they usually contain
        the most useful high-level company information.

        Args:
            company_name: e.g. "Danone"
            company_url: e.g. "https://www.danone.com"

        Returns:
            List of Documents (one per page successfully scraped)
        """
        # Common pages that contain useful company info
        paths = ["", "/about", "/about-us", "/company", "/who-we-are"]
        urls = [company_url.rstrip("/") + path for path in paths]

        # Fire all requests concurrently — this is the async advantage
        tasks = [self.scrape_url(url, company_name, doc_type="website") for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures (None values and exceptions)
        documents = [r for r in results if isinstance(r, Document)]
        logger.info(f"Scraped {len(documents)}/{len(urls)} pages from {company_url}")
        return documents

    async def scrape_company(self, company_name: str, company_url: str | None = None) -> list[Document]:
        """
        Main entry point — scrape all available sources for a company.

        Always scrapes Wikipedia. Optionally scrapes the official website.

        Args:
            company_name: Name of the company to research
            company_url: Optional official website URL

        Returns:
            All collected Documents, ready to be chunked and embedded
        """
        documents = []

        # Always start with Wikipedia — most reliable source
        logger.info(f"Starting research for: {company_name}")
        wiki_doc = await self.scrape_wikipedia(company_name)
        if wiki_doc:
            documents.append(wiki_doc)
            logger.info(f"Wikipedia: {len(wiki_doc.page_content)} characters extracted")

        # Optionally scrape official website
        if company_url:
            site_docs = await self.scrape_company_site(company_name, company_url)
            documents.extend(site_docs)

        logger.info(f"Total documents collected: {len(documents)}")
        return documents

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_text(self, html: str) -> str:
        """
        Parse HTML and extract only human-readable text.

        Process:
        1. Parse with BeautifulSoup
        2. Remove noise tags (scripts, nav, ads)
        3. Extract text from content tags only
        4. Clean up whitespace
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise — scripts, styles, navigation menus, etc.
        for tag in soup(self.NOISE_TAGS):
            tag.decompose()  # Remove from tree entirely

        # Extract text from content tags only
        texts = []
        for tag in soup.find_all(self.CONTENT_TAGS):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 20:  # Skip very short fragments
                texts.append(text)

        # Join and clean up excessive whitespace
        full_text = "\n".join(texts)
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _extract_title(self, html: str) -> str:
        """Extract the page <title> tag content."""
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else "Unknown"
