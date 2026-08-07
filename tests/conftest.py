"""
Test isolation.

WHY this runs at module import (not in a fixture): pytest imports conftest.py
before it collects test modules, and those test modules import `app.*` at the
top level — which reads settings and opens the database. A fixture would run too
late. Setting the environment here guarantees every app module comes up pointed
at a throwaway directory.

Without this, the suite wrote to the real ./data/devdocs.db and ./chroma_db,
leaving `testuser`/`dupuser`/`testadmin` behind between runs, making tests
order-dependent and letting a stale local DB disagree with CI.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="devdocs-tests-"))

os.environ["APP_ENV"] = "development"
os.environ["DB_PATH"] = str(_TMP / "users.db")
os.environ["CHROMA_PATH"] = str(_TMP / "chroma")
os.environ["JWT_SECRET"] = "test-only-secret-not-used-in-any-real-deployment"
# Explicitly blank, not merely absent: pydantic-settings also reads the real
# .env file, which does carry a GOOGLE_API_KEY. A set-but-empty environment
# variable wins over the file, so LLMManager falls back to MockProvider and the
# suite can never make a paid API call.
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"


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
