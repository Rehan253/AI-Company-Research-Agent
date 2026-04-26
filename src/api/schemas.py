"""
Pydantic schemas — request and response models for the API.

Every endpoint has a typed input and output model.
FastAPI validates requests automatically against these — bad input
returns a clear 422 error before your code even runs.

Why this matters:
  Without schemas, your API is a black box. With schemas, anyone
  integrating with your API knows exactly what to send and what
  they'll receive. FastAPI also generates /docs from these automatically.
"""

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Request models (what the client sends)
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request to ingest a company into the knowledge base."""
    company_name: str = Field(
        ...,
        description="Name of the company to research",
        examples=["Danone", "LVMH", "TotalEnergies"],
    )
    company_url: str | None = Field(
        default=None,
        description="Optional official website URL for richer data",
        examples=["https://www.danone.com"],
    )
    replace_existing: bool = Field(
        default=False,
        description="If True, delete existing data and re-ingest from scratch",
    )


class QueryRequest(BaseModel):
    """Request to ask a question about a company."""
    company_name: str = Field(
        ...,
        description="Company to query (must be ingested first, or agent will web search)",
        examples=["Danone"],
    )
    question: str = Field(
        ...,
        description="Natural language question about the company",
        examples=["What is Danone's annual revenue?", "Who is the CEO?"],
    )


# ─────────────────────────────────────────────
# Response models (what the API returns)
# ─────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response after ingesting a company."""
    success: bool
    company_name: str
    sources_scraped: int
    chunks_stored: int
    characters_processed: int
    duration_seconds: float
    errors: list[str]
    message: str


class CitationModel(BaseModel):
    """A single source citation in the answer."""
    number: int
    source: str
    title: str
    page: int | None = None
    relevance_score: float


class QueryResponse(BaseModel):
    """Response to a company research question."""
    answer: str
    citations: list[CitationModel]
    confidence: float
    tool_calls_made: int
    company_name: str
    had_enough_context: bool


class CompanyInfo(BaseModel):
    """Info about one ingested company."""
    name: str
    chunk_count: int


class CompaniesResponse(BaseModel):
    """List of all ingested companies."""
    companies: list[CompanyInfo]
    total: int


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    model: str
