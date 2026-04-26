"""
Dependency injection for FastAPI.

FastAPI's Depends() system lets you share expensive objects
(LangGraph agent, pipeline, vector store) across all requests
without recreating them on every call.

The embedding model takes ~2s to load. With dependency injection,
it loads ONCE at startup and is reused for every request.
Without it, every API call would wait 2s just to load the model.
"""

from functools import lru_cache

from src.agent.graph import CompanyResearchAgent
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_agent() -> CompanyResearchAgent:
    """
    Return the singleton CompanyResearchAgent.
    lru_cache ensures this is created only once — no matter how many
    requests come in, they all share the same agent instance.
    """
    return CompanyResearchAgent()


@lru_cache(maxsize=1)
def get_pipeline() -> IngestionPipeline:
    """Return the singleton IngestionPipeline."""
    return IngestionPipeline()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore."""
    return VectorStore()
