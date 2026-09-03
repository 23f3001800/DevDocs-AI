import json

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from tests.conftest import fresh_ip_headers, new_session_id, session_headers, session_owner

# WHY TestClient instead of a real running server?
# TestClient runs the app in-process — no network overhead,
# deterministic, fast, and works in CI without port conflicts.
client = TestClient(app)


# ── Rate limiting ────────────────────────────────────────────
def test_rate_limit_handler_is_registered():
    # Without this handler, hitting a limit 500s instead of returning 429.
    from slowapi.errors import RateLimitExceeded

    assert RateLimitExceeded in app.exception_handlers


# ── Health (public) ──────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "chunks_in_db" in r.json()


# ── Anonymous session identity ───────────────────────────────
def test_server_issues_an_httponly_session_cookie():
    isolated_client = TestClient(app)
    r = isolated_client.get("/usage", headers=fresh_ip_headers())
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert "devdocs_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie


def test_spoofed_session_header_cannot_override_a_signed_cookie():
    from app.main import _SESSION_COOKIE_NAME, make_session_token

    victim_owner = new_session_id()
    attacker_owner = new_session_id()
    victim_client = TestClient(app)
    attacker_client = TestClient(app)
    victim_client.cookies.set(_SESSION_COOKIE_NAME, make_session_token(victim_owner))
    attacker_client.cookies.set(_SESSION_COOKIE_NAME, make_session_token(attacker_owner))
    attacker_headers = {**fresh_ip_headers(), "X-Session-Id": victim_owner}
    response = attacker_client.post(
        "/ask", json={"question": "What is this project about?"}, headers=attacker_headers
    )
    assert response.status_code == 200
    conv_id = json.loads(response.text.split("event: meta\ndata: ", 1)[1].split("\n\n", 1)[0])["conversation_id"]
    assert attacker_client.get(f"/conversations/{conv_id}", headers=fresh_ip_headers()).status_code == 200
    assert victim_client.get(f"/conversations/{conv_id}", headers=fresh_ip_headers()).status_code == 404


# ── Ask endpoint ───────────────────────────────────────────────
def test_ask_question_too_short():
    headers = session_headers()
    # min_length=3 in AskRequest should reject 1-char queries
    r = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert r.status_code == 422  # Pydantic validation error


def test_ask_returns_streaming_response():
    headers = session_headers()
    r = client.post("/ask", json={"question": "What is this project about?"}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert len(r.text) > 0


def test_ask_k_out_of_range():
    headers = session_headers()
    r = client.post("/ask", json={"question": "valid question here", "k": 99}, headers=headers)
    assert r.status_code == 422


def test_ask_persists_conversation_and_returns_meta_event():
    import re

    headers = session_headers()
    r = client.post("/ask", json={"question": "What is this project about?"}, headers=headers)
    assert r.status_code == 200
    assert "event: meta" in r.text

    m = re.search(r"event: meta\ndata: (\{.*?\})\n\n", r.text)
    assert m
    conversation_id = json.loads(m.group(1))["conversation_id"]

    conv = client.get(f"/conversations/{conversation_id}", headers=headers)
    assert conv.status_code == 200
    roles = [msg["role"] for msg in conv.json()["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_ask_reuses_an_existing_conversation_without_a_new_meta_event():
    headers = session_headers()
    first = client.post("/ask", json={"question": "What is this project about?"}, headers=headers)
    import re

    conv_id = json.loads(re.search(r"event: meta\ndata: (\{.*?\})\n\n", first.text).group(1))[
        "conversation_id"
    ]

    second = client.post(
        "/ask",
        json={"question": "Tell me more", "conversation_id": conv_id},
        headers=headers,
    )
    assert second.status_code == 200
    assert "event: meta" not in second.text

    conv = client.get(f"/conversations/{conv_id}", headers=headers)
    # Two full turns (user+assistant) each = 4 messages, all in the same conversation.
    assert len(conv.json()["messages"]) == 4


def test_ask_with_unknown_conversation_id_is_404():
    headers = session_headers()
    r = client.post(
        "/ask", json={"question": "does this exist", "conversation_id": 99999999}, headers=headers
    )
    assert r.status_code == 404


# ── Free daily limit + BYOK ──────────────────────────────────
def test_ask_returns_limit_reached_after_the_free_daily_limit():
    limit = get_settings().free_daily_limit
    headers = session_headers()

    for _ in range(limit):
        r = client.post("/ask", json={"question": "using up a free question"}, headers=headers)
        assert r.status_code == 200
        assert "limit_reached" not in r.text

    blocked = client.post(
        "/ask", json={"question": "one too many free questions"}, headers=headers
    )
    assert blocked.status_code == 200  # SSE 200, error is in-band
    assert "limit_reached" in blocked.text
    assert "event: done" in blocked.text


def test_byok_header_bypasses_the_daily_limit(monkeypatch):
    # Force BYOK onto MockProvider so no real Gemini key/network is needed —
    # this only exercises the bypass logic in app.main.
    from app.llm_providers import MockProvider, llm_manager

    monkeypatch.setattr(llm_manager, "_byok_provider", lambda api_key: MockProvider())

    limit = get_settings().free_daily_limit
    headers = session_headers()
    for _ in range(limit):
        r = client.post("/ask", json={"question": "using up a free question"}, headers=headers)
        assert r.status_code == 200

    blocked = client.post("/ask", json={"question": "blocked now"}, headers=headers)
    assert "limit_reached" in blocked.text

    # Not a real credential — MockProvider is monkeypatched in above.
    not_a_real_key = "x" * 20
    byok_headers = {**headers, "X-Api-Key": not_a_real_key}
    ok = client.post("/ask", json={"question": "bypasses the limit"}, headers=byok_headers)
    assert ok.status_code == 200
    assert "limit_reached" not in ok.text


def test_usage_endpoint_reports_used_limit_remaining():
    headers = session_headers()
    r = client.get("/usage", headers=headers)
    assert r.status_code == 200
    limit = get_settings().free_daily_limit
    assert r.json() == {"used": 0, "limit": limit, "remaining": limit}

    client.post("/ask", json={"question": "this counts against the limit"}, headers=headers)

    r2 = client.get("/usage", headers=headers)
    assert r2.json() == {"used": 1, "limit": limit, "remaining": limit - 1}


def test_usage_issues_a_session_when_no_cookie_is_present():
    assert client.get("/usage", headers=fresh_ip_headers()).status_code == 200


# ── Ingest ────────────────────────────────────────────────────
def test_ingest_allowed_with_a_session_header():
    # A valid public source is queued and returns 202.
    headers = session_headers()
    r = client.post("/ingest", json={"source": "https://example.com"}, headers=headers)
    assert r.status_code == 202


def test_ingest_blocks_ssrf():
    # The SSRF guard must reject internal targets BEFORE the job is queued.
    headers = session_headers()
    r = client.post(
        "/ingest",
        json={"source": "http://169.254.169.254/latest/meta-data/"},
        headers=headers,
    )
    assert r.status_code == 400


def test_ingest_status_unknown_job_is_404():
    headers = session_headers()
    assert client.get("/ingest/does-not-exist", headers=headers).status_code == 404


def test_ingest_status_blocks_non_owner():
    """A different session must not be able to guess/reuse a job id to read
    another session's ingest details."""
    from app import main as m

    owner_headers = session_headers()
    other_headers = session_headers()

    job_id = "test-ownership-job"
    m._jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "submitted_by": session_owner(owner_headers),
    }
    try:
        assert client.get(f"/ingest/{job_id}", headers=owner_headers).status_code == 200
        assert client.get(f"/ingest/{job_id}", headers=other_headers).status_code == 403
    finally:
        m._jobs.pop(job_id, None)


# ── Source deletion (owner only) ─────────────────────────────
def test_delete_source_forbidden_for_non_owner():
    # A session that never ingested this source must not be able to purge it.
    headers = session_headers()
    r = client.request("DELETE", "/sources", json={"source": "x"}, headers=headers)
    assert r.status_code == 403


def test_delete_source_allowed_for_owner():
    from app.database import list_sources_for_user, record_source

    headers = session_headers()
    session_id = session_owner(headers)
    record_source(session_id, "https://owned-by-session.example/doc", "url", 3, "")

    r = client.request(
        "DELETE",
        "/sources",
        json={"source": "https://owned-by-session.example/doc"},
        headers=headers,
    )
    assert r.status_code == 200
    # No vector chunks were actually ingested in this test — only the DB row.
    assert r.json()["deleted_chunks"] == 0

    # The sources row itself is gone too.
    assert not any(
        s["source"] == "https://owned-by-session.example/doc"
        for s in list_sources_for_user(session_id)
    )


# ── Sources (per-session "what have I ingested") ─────────────
def test_sources_mine_lists_only_own_sources():
    from app.database import record_source

    headers = session_headers()
    session_id = session_owner(headers)
    other_session_id = new_session_id()
    record_source(session_id, "https://mine.example/doc", "url", 5, "")
    record_source(other_session_id, "https://other.example/doc", "url", 2, "")

    r = client.get("/sources/mine", headers=headers)
    assert r.status_code == 200
    sources = r.json()["sources"]
    assert all(s["owner"] == session_id for s in sources)
    assert any(s["source"] == "https://mine.example/doc" for s in sources)
    assert not any(s["source"] == "https://other.example/doc" for s in sources)


# ── PDF upload ingest ─────────────────────────────────────────
_MINI_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Size 4/Root 1 0 R>>\n%%EOF"
)


def test_ingest_upload_rejects_non_pdf():
    headers = session_headers()
    r = client.post(
        "/ingest/upload",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
        headers=headers,
    )
    assert r.status_code in (400, 415)


def test_ingest_upload_accepts_small_pdf():
    headers = session_headers()
    r = client.post(
        "/ingest/upload",
        files={"file": ("doc.pdf", _MINI_PDF, "application/pdf")},
        headers=headers,
    )
    assert r.status_code == 202
    assert "job_id" in r.json()


# ── Conversations (per-session private chat history) ─────────
def test_conversations_crud_and_cross_session_isolation():
    headers_a = session_headers()
    headers_b = session_headers()

    created = client.post("/conversations", json={"title": "My chat"}, headers=headers_a)
    assert created.status_code == 200
    conv_id = created.json()["id"]

    listed_a = client.get("/conversations", headers=headers_a)
    assert listed_a.status_code == 200
    assert any(c["id"] == conv_id for c in listed_a.json()["conversations"])

    # Another session's list must never include this conversation.
    listed_b = client.get("/conversations", headers=headers_b)
    assert not any(c["id"] == conv_id for c in listed_b.json()["conversations"])

    got = client.get(f"/conversations/{conv_id}", headers=headers_a)
    assert got.status_code == 200
    assert got.json()["messages"] == []

    # 404, not 403 — a 403 would confirm the conversation exists.
    assert client.get(f"/conversations/{conv_id}", headers=headers_b).status_code == 404
    assert client.delete(f"/conversations/{conv_id}", headers=headers_b).status_code == 404

    deleted = client.delete(f"/conversations/{conv_id}", headers=headers_a)
    assert deleted.status_code == 200
    assert client.get(f"/conversations/{conv_id}", headers=headers_a).status_code == 404


def test_ask_history_is_isolated_between_sessions():
    """Two different signed session cookies must never see each other's history."""
    import re

    headers_a = session_headers()
    headers_b = session_headers()

    r_a = client.post("/ask", json={"question": "What is this project about?"}, headers=headers_a)
    conv_id_a = json.loads(re.search(r"event: meta\ndata: (\{.*?\})\n\n", r_a.text).group(1))[
        "conversation_id"
    ]

    listed_b = client.get("/conversations", headers=headers_b)
    assert not any(c["id"] == conv_id_a for c in listed_b.json()["conversations"])
    assert client.get(f"/conversations/{conv_id_a}", headers=headers_b).status_code == 404


# ── Metrics (public — no accounts left to gate it) ────────────
def test_metrics_is_public():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "p95_latency_ms" in data
    assert "embedding_cache_hit_rate" in data
    assert "llm_calls" in data
