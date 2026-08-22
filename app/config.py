"""
Centralised, validated configuration.

WHY a single Settings object instead of scattered os.getenv() calls?
  - One place to see every knob the app has.
  - Type coercion + validation with a clear error, instead of a TypeError
    six layers deep when someone sets RETRIEVAL_INITIAL_K=twenty.
  - Production safety checks live in one validator rather than being
    duplicated across auth.py / database.py / llm_providers.py.

Precedence (highest first): real environment variables → .env file → defaults.
"""

import logging
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Sentinel value shipped in .env.example. Production refuses to start with it.
INSECURE_JWT_DEFAULT = "dev-secret-change-in-prod"


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
    gemini_model: str = "gemini-2.5-flash"
    llm_max_tokens: int = Field(1024, ge=64, le=32768)

    # ── Embeddings + vector store ────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "devdocs"
    embed_cache_max: int = Field(512, ge=0)

    # ── Retrieval knobs (tune these against scripts/run_evals.py) ──
    retrieval_initial_k: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    chunk_size_code: int = Field(1200, ge=100)
    chunk_overlap_code: int = Field(100, ge=0)
    chunk_size_text: int = Field(600, ge=100)
    chunk_overlap_text: int = Field(60, ge=0)

    # ── User database ────────────────────────────────────────
    db_path: str = "./data/devdocs.db"
    admin_password: str = ""

    # ── Auth ─────────────────────────────────────────────────
    jwt_secret: str = INSECURE_JWT_DEFAULT
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = Field(24, ge=1, le=720)

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
    # Tighter than ask/ingest on purpose: a legitimate user logs in rarely, an
    # attacker credential-stuffing /auth/login or spamming /auth/register does
    # not. Was previously unlimited on both routes.
    auth_rate_limit: str = "10/minute"

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
        """Fail closed: refuse to boot a production instance that is insecure.

        Signing tokens with the public sentinel means anyone can forge an admin
        JWT, and seeding a known admin password is the same class of hole. A
        loud crash at deploy time beats a silent breach.
        """
        if not self.is_production:
            if self.jwt_secret == INSECURE_JWT_DEFAULT:
                logging.getLogger(__name__).warning(
                    "JWT_SECRET is not set — using the insecure development default. "
                    "Do NOT use this in production."
                )
            return self

        problems = []
        if self.jwt_secret == INSECURE_JWT_DEFAULT:
            problems.append(
                "JWT_SECRET is unset (using the insecure built-in default). "
                "Generate one with `openssl rand -hex 32`."
            )
        if len(self.jwt_secret) < 32:
            problems.append("JWT_SECRET must be at least 32 characters in production.")
        if not self.admin_password:
            problems.append(
                "ADMIN_PASSWORD must be set in production to seed the admin account. "
                "Refusing to create an admin with a known default."
            )
        if not self.google_api_key:
            problems.append(
                "GOOGLE_API_KEY must be set in production. Without it the app would "
                "silently serve mock answers."
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
    "INSECURE_JWT_DEFAULT",
    "Settings",
    "configure_logging",
    "get_settings",
    "settings",
]
