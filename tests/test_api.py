from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from app.main import app

# WHY TestClient instead of a real running server?
# TestClient runs the app in-process — no network overhead,
# deterministic, fast, and works in CI without port conflicts.
client = TestClient(app)


def _get_auth_headers(username="testuser", password="testpass123", role="user"):
    """Helper: register a user and return auth headers."""
    # Try to register; if already exists, login instead
    r = client.post("/auth/register", json={"username": username, "password": password})
    if r.status_code == 409:  # user already exists
        r = client.post("/auth/login", json={"username": username, "password": password})
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _get_admin_headers():
    """Helper: get auth headers for an admin user."""
    # Create admin directly via database for testing
    from app.auth import create_token, hash_password
    from app.database import create_user, get_user

    user = get_user("testadmin")
    if not user:
        user = create_user("testadmin", hash_password("adminpass123"), role="admin")
    token = create_token(user["username"], user["role"])
    return {"Authorization": f"Bearer {token}"}


# ── Auth tests ───────────────────────────────────────────────
def test_register():
    import uuid

    unique_user = f"testuser_{uuid.uuid4().hex[:8]}"
    r = client.post("/auth/register", json={"username": unique_user, "password": "pass123456"})
    assert r.status_code == 200
    data = r.json()
    assert "token" in data
    assert data["role"] == "user"


def test_register_duplicate():
    client.post("/auth/register", json={"username": "dupuser", "password": "pass123456"})
    r = client.post("/auth/register", json={"username": "dupuser", "password": "pass123456"})
    assert r.status_code == 409


def test_login():
    client.post("/auth/register", json={"username": "loginuser", "password": "pass123456"})
    r = client.post("/auth/login", json={"username": "loginuser", "password": "pass123456"})
    assert r.status_code == 200
    assert "token" in r.json()


def test_login_wrong_password():
    client.post("/auth/register", json={"username": "wrongpwuser", "password": "pass123456"})
    r = client.post("/auth/login", json={"username": "wrongpwuser", "password": "wrongpassword"})
    assert r.status_code == 401


def test_me_endpoint():
    headers = _get_auth_headers("meuser", "pass123456")
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == "meuser"
    assert r.json()["role"] == "user"


# ── Rate limiting ────────────────────────────────────────────
def test_rate_limit_handler_is_registered():
    # Without this handler, hitting a limit 500s instead of returning 429.
    assert RateLimitExceeded in app.exception_handlers


# ── Health (public) ──────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "chunks_in_db" in r.json()


# ── Ask endpoint (requires user role) ────────────────────────
def test_ask_requires_auth():
    # No auth header → must get 401, never 200
    r = client.post("/ask", json={"question": "hello world"})
    assert r.status_code in (401, 403)


def test_ask_question_too_short():
    headers = _get_auth_headers("shortquser", "pass123456")
    # min_length=3 in AskRequest should reject 1-char queries
    r = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert r.status_code == 422  # Pydantic validation error


def test_ask_returns_streaming_response():
    headers = _get_auth_headers("askuser", "pass123456")
    r = client.post("/ask", json={"question": "What is this project about?"}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert len(r.text) > 0


def test_ask_k_out_of_range():
    headers = _get_auth_headers("kuser", "pass123456")
    r = client.post("/ask", json={"question": "valid question here", "k": 99}, headers=headers)
    assert r.status_code == 422


# ── Ingest (user or admin) ───────────────────────────────────
def test_ingest_requires_auth():
    r = client.post("/ingest", json={"source": "https://example.com"})
    assert r.status_code in (401, 403)


def test_ingest_allowed_for_user_role():
    # /ingest is open to the 'user' role — a plain user must NOT get 403.
    # (A valid public source is queued and returns 202.)
    headers = _get_auth_headers("regularuser", "pass123456")
    r = client.post("/ingest", json={"source": "https://example.com"}, headers=headers)
    assert r.status_code != 403
    assert r.status_code == 202


def test_ingest_blocks_ssrf_even_for_user():
    # The SSRF guard must reject internal targets BEFORE the job is queued —
    # and it applies to the 'user' role too, not just admins.
    headers = _get_auth_headers("ssrfuser", "pass123456")
    r = client.post(
        "/ingest",
        json={"source": "http://169.254.169.254/latest/meta-data/"},
        headers=headers,
    )
    assert r.status_code == 400


def test_ingest_status_allowed_for_user():
    # A user can poll job status; an unknown job is 404 (not 403).
    headers = _get_auth_headers("jobuser", "pass123456")
    assert client.get("/ingest/does-not-exist", headers=headers).status_code == 404


def test_ingest_status_unknown_job_is_404():
    headers = _get_admin_headers()
    assert client.get("/ingest/does-not-exist", headers=headers).status_code == 404


# ── Source deletion (admin only) ─────────────────────────────
def test_delete_source_requires_admin():
    headers = _get_auth_headers("deluser", "pass123456")
    r = client.request("DELETE", "/sources", json={"source": "x"}, headers=headers)
    assert r.status_code == 403


def test_delete_source_is_a_noop_for_unknown_source():
    headers = _get_admin_headers()
    r = client.request(
        "DELETE",
        "/sources",
        json={"source": "https://nothing.example/never-ingested"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["deleted_chunks"] == 0


# ── Logout / token revocation ────────────────────────────────
def test_logout_revokes_the_token():
    headers = _get_auth_headers("logoutuser", "pass123456")
    # Token works before logout
    assert client.get("/auth/me", headers=headers).status_code == 200

    assert client.post("/auth/logout", headers=headers).status_code == 200

    # A stateless JWT is normally valid until it expires; the denylist is what
    # makes logout real.
    after = client.get("/auth/me", headers=headers)
    assert after.status_code == 401
    assert "revoked" in after.json()["detail"].lower()


def test_logout_requires_auth():
    assert client.post("/auth/logout").status_code in (401, 403)


# ── Metrics (requires admin role) ────────────────────────────
def test_metrics_requires_admin():
    headers = _get_auth_headers("metricsuser", "pass123456")
    r = client.get("/metrics", headers=headers)
    assert r.status_code == 403  # user role can't access admin routes


def test_metrics_with_admin():
    headers = _get_admin_headers()
    r = client.get("/metrics", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "p95_latency_ms" in data
    assert "embedding_cache_hit_rate" in data
    assert "llm_calls" in data
