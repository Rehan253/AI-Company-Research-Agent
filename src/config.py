"""
Central configuration for the AI Company Research Agent.

Uses Pydantic BaseSettings to automatically load values from:
  1. Environment variables (highest priority)
  2. .env file
  3. Default values defined here (lowest priority)

This means zero code changes between local dev and production.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env file automatically
        env_file_encoding="utf-8",
        case_sensitive=False,     # OPENAI_API_KEY == openai_api_key
        extra="ignore",           # Ignore unknown env vars
    )

    # --- LLM Provider ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # --- Web Search ---
    tavily_api_key: str = ""

    # --- Vector Database ---
    chroma_persist_dir: str = "./data/chromadb"

    # --- Embeddings ---
    # all-MiniLM-L6-v2 is free, runs locally, 384-dimensional vectors
    # Good enough for our use case. OpenAI's ada-002 is better but costs money.
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    # 800 tokens per chunk = ~600 words = one or two paragraphs
    # 100 token overlap = chunks share some context at boundaries (prevents losing info)
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Logging ---
    log_level: str = "INFO"


# Single instance used everywhere — import this, not the class
settings = Settings()
