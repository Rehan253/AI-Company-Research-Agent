"""
FastAPI application entry point.

Creates the app, adds middleware, mounts routes, and configures startup.

Run with:
    uvicorn src.api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs     ← interactive API explorer (auto-generated)
    http://localhost:8000/redoc    ← alternative docs view
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.dependencies import get_agent, get_pipeline, get_vector_store
from src.api.routes import router
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.

    Code before 'yield' runs at startup — we pre-load heavy components
    so the first API request isn't slow.

    Code after 'yield' runs at shutdown — cleanup if needed.
    """
    # Pre-load all singleton components at startup
    # This loads the embedding model (~2s) before any request arrives
    logger.info("Starting AI Company Research Agent API...")
    logger.info(f"LLM model: {settings.llm_model}")
    logger.info(f"Embedding model: {settings.embedding_model}")

    get_vector_store()   # Connect to ChromaDB
    get_pipeline()       # Initialize ingestion pipeline (loads embedder)
    get_agent()          # Initialize LangGraph agent

    logger.info("All components loaded — API ready")
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="AI Company Research Agent",
    description=(
        "Autonomous AI agent for company research. "
        "Ingests company data from web and PDFs, "
        "answers questions with cited sources using RAG + LangGraph."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allows the Streamlit frontend (port 8501) to call this API (port 8000)
# Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes under /api/v1
app.include_router(router)


@app.get("/", tags=["System"])
async def root():
    """Redirect hint for the root path."""
    return {
        "message": "AI Company Research Agent API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
