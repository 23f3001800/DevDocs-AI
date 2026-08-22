"""
Test isolation.

WHY this runs at module import (not in a fixture): pytest imports conftest.py
before it collects test modules, and those test modules import `app.*` at the
top level — which reads settings and opens the database. A fixture would run too
late. Setting the environment here guarantees every app module comes up pointed
at a throwaway directory.

Without this, the suite wrote to the real ./data/devdocs.db and ./chroma_db,
leaving rows behind between runs, making tests order-dependent and letting a
stale local DB disagree with CI.
"""

import itertools
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="devdocs-tests-"))

os.environ["APP_ENV"] = "development"
os.environ["DB_PATH"] = str(_TMP / "sessions.db")
os.environ["CHROMA_PATH"] = str(_TMP / "chroma")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
# Explicitly blank, not merely absent: pydantic-settings also reads the real
# .env file, which does carry a GOOGLE_API_KEY. A set-but-empty environment
# variable wins over the file, so LLMManager falls back to MockProvider and the
# suite can never make a paid API call.
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Every request from TestClient shares one socket peer ("testclient"), so
# without this every test in the suite would fall into the SAME per-IP rate
# limit bucket for /ask and /ingest — tripping ASK_RATE_LIMIT / INGEST_RATE_LIMIT
# well before the suite finishes. Trusting X-Forwarded-For here lets tests
# that care about isolation supply a distinct fake client IP per call; tests
# that don't set the header keep behaving exactly as before.
os.environ["TRUST_PROXY_HEADERS"] = "true"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp_dir():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture
def tmp_env(monkeypatch):
    """Override settings within a single test, clearing the cached singleton."""
    from app.config import get_settings

    def _set(**overrides):
        for key, value in overrides.items():
            monkeypatch.setenv(key.upper(), str(value))
        get_settings.cache_clear()
        return get_settings()

    yield _set
    get_settings.cache_clear()


# ── Anonymous session headers ────────────────────────────────
# No login: every caller is identified by a client-generated X-Session-Id
# header. Shared helpers for the test modules that hit the API via TestClient.
_ip_counter = itertools.count(1)


def new_session_id() -> str:
    return str(uuid.uuid4())


def fresh_ip_headers() -> dict:
    """A distinct fake client IP per call, so per-IP rate limits (keyed via
    TRUST_PROXY_HEADERS=true above) don't lump every test into one bucket."""
    n = next(_ip_counter)
    return {"X-Forwarded-For": f"10.{(n >> 16) % 256}.{(n >> 8) % 256}.{n % 256}"}


def session_headers(session_id: str | None = None) -> dict:
    """Headers for a request from a fresh (or given) anonymous session."""
    headers = fresh_ip_headers()
    headers["X-Session-Id"] = session_id or new_session_id()
    return headers
