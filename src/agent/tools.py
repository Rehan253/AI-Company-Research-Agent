"""
Agent Tools — the actions the LangGraph agent can take.

Each tool is a Python function wrapped with @tool decorator.
The agent (LLM) reads the docstring to decide WHEN to use each tool.
This means docstrings here are prompts — write them clearly.

Tool selection logic (agent's reasoning):
  - vector_search:    company is ingested + question about stored facts
  - web_search:       need current/recent info OR company not ingested
  - summarize_text:   retrieved text is too long to use directly
  - ingest_company:   user wants to research a new company

Interview talking point:
  "Each tool has a carefully written description because the LLM reads
  these descriptions to decide which tool to use. Vague descriptions
  cause the agent to pick the wrong tool — so the description IS the prompt."
"""

import asyncio
import json

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from loguru import logger
from tavily import TavilyClient

from src.config import settings
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vector_store import VectorStore

# Shared instances — initialized once, reused across tool calls
# (Loading the embedding model takes ~2s — we don't want that per tool call)
_embedder = None
_vector_store = None
_tavily_client = None
_llm = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    return _llm


# ─────────────────────────────────────────────────────────────────
# TOOL 1: Vector Search
# ─────────────────────────────────────────────────────────────────

@tool
def vector_search(query: str, company_name: str) -> str:
    """
    Search the internal knowledge base for information about a company.

    Use this tool when:
    - The user asks a factual question about a company
    - The company has already been ingested into the knowledge base
    - You need specific facts, figures, history, or details

    Do NOT use this for recent news or events after the ingestion date.
    Prefer this over web_search when the company is already in the knowledge base.

    Args:
        query: The specific question or topic to search for
        company_name: The name of the company to search (e.g. "Danone", "LVMH")

    Returns:
        Relevant text chunks from the knowledge base with source references
    """
    logger.info(f"[Tool] vector_search | company={company_name} | query={query}")

    embedder = _get_embedder()
    vector_store = _get_vector_store()

    # Check if company is ingested
    stats = vector_store.get_company_stats(company_name)
    if not stats["exists"]:
        return (
            f"No data found for '{company_name}' in the knowledge base. "
            f"Use the ingest_company tool first, or use web_search for live information."
        )

    # Embed query and search
    query_embedding = embedder.embed_query(query)
    results = vector_store.query(company_name, query_embedding, n_results=5)

    if not results:
        return f"No relevant information found for query: '{query}'"

    # Format results for the agent
    output_parts = [f"Found {len(results)} relevant chunks for '{company_name}':\n"]
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("relevance_score", 0)
        output_parts.append(
            f"[{i}] (relevance={score:.2f}) Source: {source}\n{doc.page_content}\n"
        )

    return "\n".join(output_parts)


# ─────────────────────────────────────────────────────────────────
# TOOL 2: Web Search
# ─────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the live web for current information using Tavily.

    Use this tool when:
    - The user asks about recent news, latest developments, or current events
    - The company has NOT been ingested into the knowledge base
    - You need information more recent than what is stored locally
    - The vector_search returned no useful results

    Args:
        query: The search query (be specific, include company name)
               Example: "Danone Q1 2024 earnings results"
               Example: "LVMH latest acquisition 2024"

    Returns:
        Search results with titles, URLs, and content snippets
    """
    logger.info(f"[Tool] web_search | query={query}")

    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            max_results=5,
            search_depth="basic",
            include_answer=True,   # Tavily provides a synthesized answer
        )

        output_parts = []

        # Include Tavily's synthesized answer if available
        if response.get("answer"):
            output_parts.append(f"Summary: {response['answer']}\n")

        # Include individual search results
        results = response.get("results", [])
        output_parts.append(f"Found {len(results)} web results:\n")

        for i, result in enumerate(results, 1):
            output_parts.append(
                f"[{i}] {result.get('title', 'No title')}\n"
                f"URL: {result.get('url', '')}\n"
                f"{result.get('content', '')[:400]}\n"
            )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search failed: {str(e)}. Try a different query."


# ─────────────────────────────────────────────────────────────────
# TOOL 3: Summarize Text
# ─────────────────────────────────────────────────────────────────

@tool
def summarize_text(text: str, focus: str = "") -> str:
    """
    Summarize a long piece of text into a concise paragraph.

    Use this tool when:
    - You retrieved a very long document or chunk that needs condensing
    - You want to extract key points from raw scraped content
    - The user asks for a summary of a document or section

    Args:
        text: The text to summarize (can be long)
        focus: Optional — what aspect to focus on in the summary
               Example: "financial performance", "leadership team", "products"

    Returns:
        A concise summary paragraph (3-5 sentences)
    """
    logger.info(f"[Tool] summarize_text | length={len(text)} | focus='{focus}'")

    if len(text) < 200:
        return text  # Already short enough, no need to summarize

    focus_instruction = f" Focus specifically on: {focus}." if focus else ""
    prompt = (
        f"Summarize the following text in 3-5 concise sentences.{focus_instruction}\n\n"
        f"Text:\n{text[:4000]}"  # Cap at 4000 chars to stay within token limits
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return text[:500] + "..."  # Fallback: return truncated original


# ─────────────────────────────────────────────────────────────────
# TOOL 4: Ingest Company
# ─────────────────────────────────────────────────────────────────

@tool
def ingest_company(company_name: str, company_url: str = "") -> str:
    """
    Ingest a company into the knowledge base by scraping and storing its data.

    Use this tool when:
    - The user asks to research a company not yet in the knowledge base
    - vector_search returns "No data found" for a company
    - The user explicitly asks to "add", "ingest", or "learn about" a company

    This tool scrapes Wikipedia and optionally the company website,
    chunks the text, generates embeddings, and stores everything in ChromaDB.
    After this tool runs, vector_search will work for this company.

    Args:
        company_name: Name of the company to ingest (e.g. "TotalEnergies")
        company_url: Optional official website URL for richer data

    Returns:
        Ingestion report with number of chunks stored and time taken
    """
    logger.info(f"[Tool] ingest_company | company={company_name} | url={company_url}")

    try:
        pipeline = IngestionPipeline()
        url = company_url if company_url else None

        # Run the async pipeline synchronously inside the sync tool
        # The agent framework is sync, but our pipeline is async
        report = asyncio.run(
            pipeline.ingest_company(company_name, company_url=url)
        )

        if report.success:
            return (
                f"Successfully ingested '{company_name}':\n"
                f"  - Sources scraped: {report.sources_scraped}\n"
                f"  - Chunks stored: {report.total_chunks}\n"
                f"  - Characters processed: {report.total_characters:,}\n"
                f"  - Time: {report.duration_seconds}s\n"
                f"You can now use vector_search to answer questions about {company_name}."
            )
        else:
            return (
                f"Ingestion of '{company_name}' had issues:\n"
                f"  Errors: {report.errors}\n"
                f"  Chunks stored: {report.total_chunks}\n"
                f"Try web_search as a fallback."
            )
    except Exception as e:
        logger.error(f"Ingestion tool failed: {e}")
        return f"Failed to ingest '{company_name}': {str(e)}"


# ─────────────────────────────────────────────────────────────────
# Export all tools as a list for the agent
# ─────────────────────────────────────────────────────────────────

AGENT_TOOLS = [vector_search, web_search, summarize_text, ingest_company]
