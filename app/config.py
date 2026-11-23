"""Validated environment settings; construction does not create files or call providers."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment overrides .env; explicit constructor values override both."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )
    data_dir: Path = Path("data")
    max_upload_mb: int = Field(default=25, ge=1, le=1024)
    max_archive_mb: int = Field(default=100, ge=1, le=2048)
    max_extracted_chars: int = Field(default=5_000_000, ge=1)
    max_pdf_pages: int = Field(default=1000, ge=1)
    max_table_rows: int = Field(default=100_000, ge=1)
    table_rows_per_block: int = Field(default=20, ge=1, le=1000)
    chunk_size: int = Field(default=1000, ge=32)
    chunk_overlap: int = Field(default=150, ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ocr_enabled: bool = False
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=200, ge=72, le=600)
    ocr_timeout_seconds: int = Field(default=60, ge=1, le=600)
    poppler_path: str | None = None
    tesseract_cmd: str | None = None
    embedding_provider: Literal["local", "ollama", "gemini", "openai"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: Literal["ollama", "gemini", "openrouter", "openai", "anthropic", "groq"] = (
        "openrouter"
    )
    llm_model: str = "google/gemma-4-31b-it:free"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.25, ge=-1, le=1)
    collection: str = "documents"
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def validate_paths_and_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    """Load once per process; call cache_clear() to reload environment defaults."""
    return Settings()
