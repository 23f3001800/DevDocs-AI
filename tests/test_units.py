"""
Fast, offline unit tests for DevDocs AI.

These cover the pure logic — BM25, chunking, response schema, provider
fallback, the SSRF guard, and the embedding-cache stats — WITHOUT downloading
models or calling any paid LLM API. They run in seconds and need no secrets,
so they make a good CI floor.

Run just these:   pytest tests/test_units.py -v
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from pydantic import ValidationError

from app.bm25_retriever import BM25Retriever
from app.chunker import chunk_documents
from app.llm_providers import GeminiProvider, LLMManager, LLMResponse, MockProvider
from app.main import validate_ingest_source
from app.models import RAGResponse
from app.vectorstore import get_cache_stats

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────
# config.py — production fail-fast without a Gemini key
# ─────────────────────────────────────────────────────────────
def _import_config_with_env(**overrides):
    """Import app.config in a clean subprocess with the given env overrides."""
    env = dict(os.environ)
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", "import app.config"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_config_refuses_production_without_a_google_api_key():
    result = _import_config_with_env(APP_ENV="production", GOOGLE_API_KEY="")
    assert result.returncode != 0
    assert "GOOGLE_API_KEY" in result.stderr


def test_config_refuses_production_without_a_session_secret():
    result = _import_config_with_env(APP_ENV="production", GOOGLE_API_KEY="test-key", SESSION_SECRET="")
    assert result.returncode != 0
    assert "SESSION_SECRET" in result.stderr


def test_config_allows_development_without_a_google_api_key():
    result = _import_config_with_env(APP_ENV="development", GOOGLE_API_KEY="")
    assert result.returncode == 0


# ─────────────────────────────────────────────────────────────
# bm25_retriever.py
# ─────────────────────────────────────────────────────────────
DOCS = [
    "FastAPI is a modern web framework for building APIs with Python.",
    "BM25 ranks documents by term frequency and inverse document frequency.",
    "ChromaDB stores vector embeddings for semantic search.",
]
METAS = [{"file_path": f"doc{i}.md"} for i in range(len(DOCS))]
IDS = [f"id{i}" for i in range(len(DOCS))]


def test_bm25_defaults_ids_when_omitted():
    r = BM25Retriever()
    r.build_index(DOCS, METAS)
    hits = r.search("BM25 term frequency", k=1)
    assert hits and hits[0]["id"] == "1"


def test_bm25_empty_index_returns_empty():
    assert BM25Retriever().search("anything") == []


def test_bm25_finds_relevant_doc_first():
    r = BM25Retriever()
    r.build_index(DOCS, METAS, IDS)
    results = r.search("BM25 term frequency", k=3)
    assert results, "expected at least one hit"
    top = results[0]
    assert "BM25" in top["content"]
    assert top["metadata"]["file_path"] == "doc1.md"
    assert top["id"] == "id1"
    assert top["score"] > 0


def test_bm25_respects_k_limit():
    r = BM25Retriever()
    r.build_index(DOCS, METAS, IDS)
    assert len(r.search("python api framework", k=1)) <= 1


def test_bm25_no_longer_hard_gates_on_positive_score():
    """The old `if scores[i] > 0` check dropped legitimate matches whenever
    BM25's IDF went non-positive (common in small/narrow corpora) — ranking is
    BM25's job, relevance filtering belongs to the downstream reranker."""
    r = BM25Retriever()
    r.build_index(DOCS, METAS, IDS)
    # Every document scores 0 against these terms, but search() still returns
    # up to k results ranked by score, rather than hard-filtering them out.
    results = r.search("zzzz qqqq xxxx", k=2)
    assert len(results) == 2
    assert all(hit["score"] == 0.0 for hit in results)


# ─────────────────────────────────────────────────────────────
# chunker.py
# ─────────────────────────────────────────────────────────────
def test_chunker_adds_index_metadata():
    doc = Document(
        page_content="First paragraph.\n\n" + ("word " * 300),
        metadata={"file_path": "notes.md", "file_type": "md"},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_index" in c.metadata
        assert "total_chunks" in c.metadata
        assert c.metadata["file_path"] == "notes.md"  # original metadata preserved


def test_chunker_splits_long_document():
    long_doc = Document(
        page_content="para. " * 2000,  # well over the 600-char doc chunk size
        metadata={"file_type": "md"},
    )
    chunks = chunk_documents([long_doc])
    assert len(chunks) > 1


def test_chunker_handles_code_language():
    code = "def foo():\n    return 1\n\n" * 100
    doc = Document(page_content=code, metadata={"file_type": "py"})
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1
    assert all("chunk_index" in c.metadata for c in chunks)


# ─────────────────────────────────────────────────────────────
# models.py — RAGResponse schema validation
# ─────────────────────────────────────────────────────────────
def test_ragresponse_valid():
    r = RAGResponse(answer="hi", sources=["a.py"], confidence=0.8, has_answer=True)
    assert r.confidence == 0.8
    assert r.sources == ["a.py"]


def test_ragresponse_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        RAGResponse(answer="x", sources=[], confidence=1.5, has_answer=True)
    with pytest.raises(ValidationError):
        RAGResponse(answer="x", sources=[], confidence=-0.1, has_answer=True)


# ─────────────────────────────────────────────────────────────
# llm_providers.py — MockProvider + fallback logic
# ─────────────────────────────────────────────────────────────
def test_mock_provider_generate():
    resp = MockProvider().generate("sys", "user")
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "mock"


def _fake_provider(name, *, fail=False):
    class _P:
        def __init__(self):
            self.name = name
            self.model = f"{name}-model"

        def generate(self, system, user_message, max_tokens=1024):
            if fail:
                raise RuntimeError(f"{name} boom")
            return LLMResponse(text=f"ok-{name}", provider=name, model=self.model)

        async def stream(self, system, user_message, max_tokens=1024):
            if fail:
                raise RuntimeError(f"{name} stream boom")
            yield f"ok-{name}"

    return _P()


def _manager_with(providers):
    mgr = LLMManager.__new__(LLMManager)  # bypass __init__ / env detection
    mgr.providers = providers
    return mgr


def test_manager_falls_back_to_next_provider():
    mgr = _manager_with([_fake_provider("primary", fail=True), _fake_provider("backup")])
    resp = mgr.generate("sys", "user")
    assert resp.provider == "backup"
    assert resp.text == "ok-backup"


def test_manager_uses_primary_when_healthy():
    mgr = _manager_with([_fake_provider("primary"), _fake_provider("backup")])
    assert mgr.generate("sys", "user").provider == "primary"


def test_manager_raises_when_all_fail():
    mgr = _manager_with([_fake_provider("a", fail=True), _fake_provider("b", fail=True)])
    with pytest.raises(RuntimeError) as exc:
        mgr.generate("sys", "user")
    assert "All LLM providers failed" in str(exc.value)


def test_manager_provider_names():
    mgr = _manager_with([_fake_provider("x"), _fake_provider("y")])
    assert mgr.provider_names() == ["x", "y"]
    assert mgr.active_provider == "x"


async def test_manager_stream_falls_back():
    mgr = _manager_with([_fake_provider("primary", fail=True), _fake_provider("backup")])
    out = "".join([tok async for tok in mgr.stream("sys", "user")])
    assert out == "ok-backup"


async def test_mock_provider_stream():
    out = "".join([tok async for tok in MockProvider().stream("sys", "user")])
    assert "Mock response" in out


# ── BYOK: per-request Gemini key, no shared/global state ──────
_NOT_A_REAL_KEY = "n" + "ot-a-real-key"  # not a credential — client is mocked


def test_gemini_provider_byok_uses_the_caller_key_not_the_server_key(monkeypatch):
    from google import genai as genai_mod

    seen = {}

    class _FakeClient:
        def __init__(self, api_key):
            seen["api_key"] = api_key

    monkeypatch.setattr(genai_mod, "Client", _FakeClient)
    GeminiProvider(api_key=_NOT_A_REAL_KEY)
    assert seen["api_key"] == _NOT_A_REAL_KEY


def test_manager_byok_provider_builds_a_gemini_provider_with_the_given_key(monkeypatch):
    mgr = LLMManager.__new__(LLMManager)
    seen = {}
    monkeypatch.setattr(
        "app.llm_providers.GeminiProvider",
        lambda api_key=None: seen.setdefault("api_key", api_key),
    )
    mgr._byok_provider(_NOT_A_REAL_KEY)
    assert seen["api_key"] == _NOT_A_REAL_KEY


def test_manager_generate_with_api_key_bypasses_the_provider_chain(monkeypatch):
    """A BYOK call must never fall through to the server's own providers —
    the caller's key succeeding or failing is entirely on its own."""
    mgr = _manager_with([_fake_provider("primary", fail=True)])
    monkeypatch.setattr(mgr, "_byok_provider", lambda api_key: _fake_provider("byok"))
    resp = mgr.generate("sys", "user", api_key=_NOT_A_REAL_KEY)
    assert resp.provider == "byok"


async def test_manager_stream_with_api_key_bypasses_the_provider_chain(monkeypatch):
    mgr = _manager_with([_fake_provider("primary", fail=True)])
    monkeypatch.setattr(mgr, "_byok_provider", lambda api_key: _fake_provider("byok"))
    out = "".join([tok async for tok in mgr.stream("sys", "user", api_key=_NOT_A_REAL_KEY)])
    assert out == "ok-byok"


# ─────────────────────────────────────────────────────────────
# main.py — SSRF guard for /ingest sources
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "bad",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1:8000/admin",  # loopback
        "http://localhost/internal",  # loopback by name
        "http://10.0.0.5/",  # private range
        "ftp://example.com/file",  # disallowed scheme
        "file:///etc/passwd",  # local file scheme
    ],
)
def test_validate_ingest_source_rejects_internal(bad):
    with pytest.raises(HTTPException):
        validate_ingest_source(bad)


def test_validate_ingest_source_allows_public_and_local():
    # Public host and non-URL local paths should pass without raising.
    validate_ingest_source("https://github.com/tiangolo/fastapi")
    validate_ingest_source("/data/manual.pdf")


# ─────────────────────────────────────────────────────────────
# main.py — anonymous session identity (get_session_id)
# ─────────────────────────────────────────────────────────────
class _FakeRequest:
    """Minimal stand-in exposing just what get_session_id reads."""

    def __init__(self, session_id: str | None = None):
        self.state = SimpleNamespace(session_id=session_id)


def test_get_session_id_requires_verified_middleware_state():
    from app.main import get_session_id

    with pytest.raises(HTTPException) as exc:
        get_session_id(_FakeRequest())
    assert exc.value.status_code == 401


def test_get_session_id_reads_only_the_verified_owner():
    from app.main import get_session_id

    raw = "550e8400-e29b-41d4-a716-446655440000"
    assert get_session_id(_FakeRequest(raw)) == raw


def test_session_token_round_trip_and_tamper_rejection():
    from app.main import make_session_token, session_id_from_token

    raw = "550e8400-e29b-41d4-a716-446655440000"
    token = make_session_token(raw)
    assert session_id_from_token(token) == raw
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert session_id_from_token(tampered) is None


# ─────────────────────────────────────────────────────────────
# database.py — per-session daily usage counter
# ─────────────────────────────────────────────────────────────
def test_usage_counter_starts_at_zero_and_increments():
    import uuid

    from app.database import get_usage_count, increment_usage

    # A fresh, random session id keeps this independent of every other test
    # sharing the same on-disk test database (see tests/conftest.py).
    session_id = f"usage-test-{uuid.uuid4().hex}"
    assert get_usage_count(session_id, day="2026-01-01") == 0
    assert increment_usage(session_id, day="2026-01-01") == 1
    assert increment_usage(session_id, day="2026-01-01") == 2
    assert get_usage_count(session_id, day="2026-01-01") == 2
    # A different day is a completely independent counter.
    assert get_usage_count(session_id, day="2026-01-02") == 0


# ─────────────────────────────────────────────────────────────
# main.py — job store eviction must never drop in-flight work
# ─────────────────────────────────────────────────────────────
def test_evict_completed_jobs_never_drops_queued_or_running():
    """The bug: plain FIFO (`_jobs.popitem(last=False)`) evicted the oldest
    entry regardless of status, which could be a job that hadn't run yet —
    silently discarding it. Eviction must only ever remove terminal jobs."""
    from app import main as m

    saved = dict(m._jobs)
    m._jobs.clear()
    try:
        for i in range(m._JOBS_MAX):
            m._jobs[f"queued-{i}"] = {"job_id": f"queued-{i}", "status": "queued"}
        m._jobs["running-0"] = {"job_id": "running-0", "status": "running"}
        m._jobs["done-0"] = {"job_id": "done-0", "status": "succeeded"}
        m._jobs["done-1"] = {"job_id": "done-1", "status": "failed"}

        m._evict_completed_jobs()

        # Only the two terminal jobs were eligible, so exactly they were
        # dropped — every queued/running job survives, even over the cap.
        assert "done-0" not in m._jobs
        assert "done-1" not in m._jobs
        assert "running-0" in m._jobs
        assert all(f"queued-{i}" in m._jobs for i in range(m._JOBS_MAX))
    finally:
        m._jobs.clear()
        m._jobs.update(saved)


def test_evict_completed_jobs_is_a_noop_under_the_cap():
    from app import main as m

    saved = dict(m._jobs)
    m._jobs.clear()
    try:
        m._jobs["a"] = {"job_id": "a", "status": "succeeded"}
        m._evict_completed_jobs()
        assert "a" in m._jobs  # under _JOBS_MAX — nothing to evict yet
    finally:
        m._jobs.clear()
        m._jobs.update(saved)


# ─────────────────────────────────────────────────────────────
# vectorstore.py — embedding cache stats (no model needed)
# ─────────────────────────────────────────────────────────────
def test_cache_stats_shape_and_bounds():
    stats = get_cache_stats()
    for key in (
        "embedding_cache_hits",
        "embedding_cache_misses",
        "embedding_cache_hit_rate",
        "embedding_cache_size",
    ):
        assert key in stats
    assert 0.0 <= stats["embedding_cache_hit_rate"] <= 1.0


def test_embedder_is_a_shared_singleton(monkeypatch):
    # The real model is never loaded here — we swap in a dummy and assert the
    # loader is invoked at most once, then reused (the perf fix's contract).
    import app.vectorstore as vs

    load_count = {"n": 0}

    class _DummyModel:
        pass

    def _fake_ctor(_name):
        load_count["n"] += 1
        return _DummyModel()

    monkeypatch.setattr(vs, "SentenceTransformer", _fake_ctor)
    monkeypatch.setattr(vs, "_embedder", None)  # auto-reverted by monkeypatch

    first = vs._get_embedder()
    second = vs._get_embedder()
    assert first is second
    assert load_count["n"] == 1
