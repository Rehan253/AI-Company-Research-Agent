"""
API Routes — all endpoint definitions.

Each endpoint:
  1. Validates input (Pydantic does this automatically)
  2. Calls the right component (agent, pipeline, vector store)
  3. Returns a typed response model

Error handling:
  - 404: Company not found
  - 422: Invalid request (automatic from Pydantic)
  - 500: Internal error (caught and returned cleanly)
"""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.api.dependencies import get_agent, get_pipeline, get_vector_store
from src.api.schemas import (
    CitationModel,
    CompaniesResponse,
    CompanyInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import settings

router = APIRouter(prefix="/api/v1")


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API is running."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=settings.llm_model,
    )


# ─────────────────────────────────────────────
# Company ingestion
# ─────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_company(
    request: IngestRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Ingest a company into the knowledge base.

    Scrapes Wikipedia + optional website, chunks text, generates embeddings,
    and stores everything in ChromaDB. Takes 5-15 seconds.

    After ingestion, the company can be queried via /query.
    """
    logger.info(f"POST /ingest | company={request.company_name}")

    try:
        report = await pipeline.ingest_company(
            company_name=request.company_name,
            company_url=request.company_url,
            replace_existing=request.replace_existing,
        )

        return IngestResponse(
            success=report.success,
            company_name=report.company_name,
            sources_scraped=report.sources_scraped,
            chunks_stored=report.total_chunks,
            characters_processed=report.total_characters,
            duration_seconds=report.duration_seconds,
            errors=report.errors,
            message=(
                f"Successfully ingested {report.total_chunks} chunks for '{request.company_name}'"
                if report.success
                else f"Ingestion failed: {report.errors}"
            ),
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Query / Q&A
# ─────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["Research"])
async def query_company(
    request: QueryRequest,
    agent=Depends(get_agent),
):
    """
    Ask a question about a company using the autonomous agent.

    The agent will:
    1. Search the knowledge base (if company is ingested)
    2. Search the web for recent information if needed
    3. Return a cited answer

    No ingestion required — the agent falls back to web search automatically.
    """
    logger.info(f"POST /query | company={request.company_name} | q={request.question}")

    try:
        result = agent.research(
            question=request.question,
            company_name=request.company_name,
        )

        # Build citation models from the agent result messages
        citations = _extract_citations(result["messages"])

        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            confidence=_estimate_confidence(result["messages"]),
            tool_calls_made=result["tool_calls_made"],
            company_name=request.company_name,
            had_enough_context="don't have" not in result["answer"].lower(),
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Company management
# ─────────────────────────────────────────────

@router.get("/companies", response_model=CompaniesResponse, tags=["Management"])
async def list_companies(vector_store=Depends(get_vector_store)):
    """List all companies currently in the knowledge base."""
    company_names = vector_store.list_companies()

    companies = []
    for name in company_names:
        stats = vector_store.get_company_stats(name)
        companies.append(CompanyInfo(
            name=name,
            chunk_count=stats.get("chunk_count", 0),
        ))

    return CompaniesResponse(companies=companies, total=len(companies))


@router.delete("/companies/{company_name}", tags=["Management"])
async def delete_company(
    company_name: str,
    vector_store=Depends(get_vector_store),
):
    """Remove a company and all its data from the knowledge base."""
    stats = vector_store.get_company_stats(company_name)
    if not stats["exists"]:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{company_name}' not found in knowledge base",
        )

    success = vector_store.delete_company(company_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete company")

    return {"message": f"Company '{company_name}' deleted successfully"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_citations(messages: list) -> list[CitationModel]:
    """Extract citation info from agent messages."""
    citations = []
    seen_sources = set()
    citation_number = 1

    for msg in messages:
        # ToolMessages contain tool results with source metadata
        if hasattr(msg, "content") and "[1]" in str(msg.content):
            # Simple extraction: parse "Source: URL" lines from tool output
            for line in str(msg.content).split("\n"):
                if "Source:" in line and "http" in line:
                    source = line.split("Source:")[-1].strip()
                    if source not in seen_sources:
                        seen_sources.add(source)
                        citations.append(CitationModel(
                            number=citation_number,
                            source=source,
                            title=source.split("/")[-1] or source,
                            relevance_score=0.6,
                        ))
                        citation_number += 1

    return citations


def _estimate_confidence(messages: list) -> float:
    """Estimate confidence from the number of tool results found."""
    tool_results = [m for m in messages if hasattr(m, "type") and m.type == "tool"]
    if not tool_results:
        return 0.3
    return min(0.5 + len(tool_results) * 0.1, 0.9)
