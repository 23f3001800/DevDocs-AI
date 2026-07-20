"""
Fast, offline unit tests for DevDocs AI.

These cover the pure logic — auth/RBAC, JWT, BM25, chunking, response schema,
provider fallback, the SSRF guard, and the embedding-cache stats — WITHOUT
downloading models or calling any paid LLM API. They run in seconds and need no
secrets, so they make a good CI floor.

Run just these:   pytest tests/test_units.py -v
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from pydantic import ValidationError

from app import auth
from app.bm25_retriever import BM25Retriever
from app.chunker import chunk_documents
from app.llm_providers import LLMManager, LLMResponse, MockProvider
from app.main import validate_ingest_source
from app.models import RAGResponse
from app.vectorstore import get_cache_stats


# ─────────────────────────────────────────────────────────────
# auth.py — password hashing
# ─────────────────────────────────────────────────────────────
def test_password_hash_roundtrip():
    hashed = auth.hash_password("s3cret-pw")
    assert hashed != "s3cret-pw"  # never store plaintext
    assert auth.verify_password("s3cret-pw", hashed) is True
    assert auth.verify_password("wrong-pw", hashed) is False


def test_password_hash_is_salted():
    # bcrypt salts each hash — same input, different output
    assert auth.hash_password("same") != auth.hash_password("same")


# ─────────────────────────────────────────────────────────────
# auth.py — JWT create / decode
# ─────────────────────────────────────────────────────────────
def test_jwt_roundtrip_preserves_claims():
    token = auth.create_token("alice", "admin")
    payload = auth.decode_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"


def test_jwt_expired_is_rejected():
    expired = jwt.encode(
        {
            "sub": "bob",
            "role": "user",
            "exp": datetime.now(UTC) - timedelta(hours=1),
        },
        auth.JWT_SECRET,
        algorithm=auth.JWT_ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc:
        auth.decode_token(expired)
    assert exc.value.status_code == 401


def test_jwt_tampered_is_rejected():
    token = auth.create_token("carol", "user")
    with pytest.raises(HTTPException) as exc:
        auth.decode_token(token + "tampered")
    assert exc.value.status_code == 401


# ─────────────────────────────────────────────────────────────
# auth.py — RBAC role hierarchy
# ─────────────────────────────────────────────────────────────
def test_require_role_admin_allows_admin():
    dep = auth.require_role("admin")
    assert dep(user={"username": "a", "role": "admin"})["role"] == "admin"


def test_require_role_admin_blocks_user():
    dep = auth.require_role("admin")
    with pytest.raises(HTTPException) as exc:
        dep(user={"username": "u", "role": "user"})
    assert exc.value.status_code == 403


def test_require_role_user_allows_admin():
    # admin >= user in the hierarchy, so admin can access user routes
    dep = auth.require_role("user")
    assert dep(user={"username": "a", "role": "admin"})["role"] == "admin"


def test_require_role_unknown_role_is_denied():
    dep = auth.require_role("user")
    with pytest.raises(HTTPException):
        dep(user={"username": "x", "role": "nonsense"})


# ─────────────────────────────────────────────────────────────
# bm25_retriever.py
# ─────────────────────────────────────────────────────────────
DOCS = [
    "FastAPI is a modern web framework for building APIs with Python.",
    "BM25 ranks documents by term frequency and inverse document frequency.",
    "ChromaDB stores vector embeddings for semantic search.",
]
METAS = [{"file_path": f"doc{i}.md"} for i in range(len(DOCS))]


def test_bm25_empty_index_returns_empty():
    assert BM25Retriever().search("anything") == []


def test_bm25_finds_relevant_doc_first():
    r = BM25Retriever()
    r.build_index(DOCS, METAS)
    results = r.search("BM25 term frequency", k=3)
    assert results, "expected at least one hit"
    top_doc, top_meta, top_score = results[0]
    assert "BM25" in top_doc
    assert top_meta["file_path"] == "doc1.md"
    assert top_score > 0


def test_bm25_respects_k_limit():
    r = BM25Retriever()
    r.build_index(DOCS, METAS)
    assert len(r.search("python api framework", k=1)) <= 1


def test_bm25_filters_zero_scores():
    r = BM25Retriever()
    r.build_index(DOCS, METAS)
    # A query with no overlapping terms should return nothing (scores are 0)
    assert r.search("zzzz qqqq xxxx", k=3) == []


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
