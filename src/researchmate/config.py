"""Application configuration.

All tunable values live here and are read from environment variables or the
project's ``.env`` file. Business logic never reads ``os.environ`` directly; it
receives a :class:`Settings` instance instead, which keeps configuration
separate from behaviour and makes components easy to test with custom values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/researchmate/config.py -> parents[2] is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed, validated application settings.

    Field names map to upper-case environment variables (``groq_model`` <-
    ``GROQ_MODEL``). Values are validated on construction, so a typo such as
    ``CHUNK_SIZE_TOKENS=abc`` fails at startup instead of deep inside a request.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "ResearchMate"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    # --- Groq LLM (OpenAI-compatible API) --------------------------------
    # SecretStr hides the value in repr()/str()/logs; call .get_secret_value()
    # only at the single place where the HTTP client is created.
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    # Used when GROQ_MODEL is not listed by the live /models endpoint.
    groq_fallback_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    database_url: str | None = None

    # --- Local models ------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_device: Literal["cpu", "cuda", "auto"] = "cpu"

    # --- Ingestion -------------------------------------------------------
    max_upload_mb: int = Field(default=25, gt=0, le=200)
    chunk_size_tokens: int = Field(default=400, ge=50, le=512)
    chunk_overlap_tokens: int = Field(default=60, ge=0)

    # --- Retrieval -------------------------------------------------------
    dense_top_k: int = Field(default=20, ge=1, le=200)
    bm25_top_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_n: int = Field(default=6, ge=1, le=50)

    # ------------------------------------------------------------------
    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        """Treat ``GROQ_API_KEY=`` (empty) the same as "not configured"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def _empty_url_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        """Resolve relative paths against the project root, not the CWD.

        Without this, starting the API from another folder would silently
        create a second, empty database somewhere else.
        """
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS "
                f"(got {self.chunk_overlap_tokens} >= {self.chunk_size_tokens})"
            )
        return self

    # --- Derived paths (single source of truth for on-disk layout) ---------
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def hf_cache_dir(self) -> Path:
        return self.data_dir / "hf_cache"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "researchmate.db"

    @property
    def resolved_database_url(self) -> str:
        """``DATABASE_URL`` if set, otherwise the local SQLite file."""
        return self.database_url or f"sqlite:///{self.sqlite_path.as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def groq_configured(self) -> bool:
        return self.groq_api_key is not None


def ensure_directories(settings: Settings) -> None:
    """Create the runtime data folders if they do not exist yet."""
    for path in (settings.data_dir, settings.uploads_dir, settings.chroma_dir,
                 settings.hf_cache_dir):
        path.mkdir(parents=True, exist_ok=True)


def configure_runtime_environment(settings: Settings) -> None:
    """Set process-level environment variables that third-party libraries read.

    Must run **before** importing ``sentence_transformers`` / ``huggingface_hub``
    / ``chromadb``, because they read these variables at import time.

    * ``HF_HOME`` keeps downloaded models on D: (inside the project) instead of
      the default ``C:\\Users\\<user>\\.cache\\huggingface``.
    * ``ANONYMIZED_TELEMETRY=False`` disables ChromaDB's usage telemetry.

    ``setdefault`` is used so a value set explicitly in the shell still wins.
    """
    os.environ.setdefault("HF_HOME", str(settings.hf_cache_dir))
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
