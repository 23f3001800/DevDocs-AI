"""
Centralised, validated configuration.

WHY a single Settings object instead of scattered os.getenv() calls?
  - One place to see every knob the app has.
  - Type coercion + validation with a clear error, instead of a TypeError
    six layers deep when someone sets RETRIEVAL_INITIAL_K=twenty.
  - Production safety checks live in one validator rather than being
    duplicated across llm_providers.py and anywhere else that cares.

Precedence (highest first): real environment variables → .env file → defaults.
"""

import logging
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env carries keys we no longer read (API_KEY, LANGCHAIN_*)
    )

    # ── Runtime ──────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── LLM (Gemini is the only provider) ────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    llm_max_tokens: int = Field(1024, ge=64, le=32768)

    # ── Embeddings + vector store ────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "devdocs"
    embed_cache_max: int = Field(512, ge=0)

    # ── Retrieval knobs ──────────────────────────────────────
    retrieval_initial_k: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    chunk_size_code: int = Field(1200, ge=100)
    chunk_overlap_code: int = Field(100, ge=0)
    chunk_size_text: int = Field(600, ge=100)
    chunk_overlap_text: int = Field(60, ge=0)

    # ── Session database ─────────────────────────────────────
    db_path: str = "./data/devdocs.db"

    # ── Uploads ──────────────────────────────────────────────
    upload_dir: str = "./data/uploads"

    # Free questions per calendar day (UTC) per signed session cookie, absent
    # an X-Api-Key — see the `usage` table.
    free_daily_limit: int = Field(5, ge=0)
    # HMAC signing key for the anonymous session cookie. Development may use a
    # process-local fallback so a checkout works out of the box; production
    # must set a durable secret or every session would be invalidated on restart.
    session_secret: str = ""
    session_max_age_seconds: int = Field(60 * 60 * 24 * 30, ge=300, le=60 * 60 * 24 * 365)
    session_cookie_secure: bool = False

    # ── HTTP ─────────────────────────────────────────────────
    allowed_origins: str = "*"
    # slowapi/limits storage. "memory://" is per-process — set a redis:// URI
    # when running more than one worker or replica, or the limit is multiplied
    # by the number of processes.
    rate_limit_storage_uri: str = "memory://"
    # Only trust X-Forwarded-For when you are actually behind a proxy you
    # control. Otherwise a client can spoof the header and dodge rate limits.
    trust_proxy_headers: bool = False
    ask_rate_limit: str = "30/minute"
    ingest_rate_limit: str = "5/minute"

    # ── Ingestion ────────────────────────────────────────────
    ingest_max_pages: int = Field(20, ge=1)
    ingest_timeout_seconds: int = Field(10, ge=1)

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> "Settings":
        """Refuse to boot in production with no LLM key — better a loud crash
        at deploy time than silently serving mock answers behind /health."""
        if not self.is_production:
            return self

        problems = []
        if not self.google_api_key:
            problems.append(
                "GOOGLE_API_KEY must be set in production. Without it the app would "
                "silently serve mock answers."
            )
        if len(self.session_secret) < 32:
            problems.append(
                "SESSION_SECRET must be at least 32 characters in production to sign session cookies."
            )
        if problems:
            raise ValueError(
                "Refusing to start with APP_ENV=production:\n  - " + "\n  - ".join(problems)
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Cached so validation runs once."""
    return Settings()


def configure_logging() -> None:
    """Set up root logging once, honouring LOG_LEVEL.

    WHY not print()? print has no levels, no timestamps, cannot be filtered or
    shipped to a log aggregator, and writes to stdout unconditionally.
    """
    level = getattr(logging, get_settings().log_level.strip().upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    # These libraries are extremely chatty at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


# Importing this module must be enough to trigger the production safety check,
# so that a misconfigured deploy dies at boot rather than on first request.
settings = get_settings()

__all__ = [
    "Settings",
    "configure_logging",
    "get_settings",
    "settings",
]
