import pytest
from fastapi.testclient import TestClient
from app.main import app

# WHY TestClient instead of a real running server?
# TestClient runs the app in-process — no network overhead,
# deterministic, fast, and works in CI without port conflicts.
client = TestClient(app)


def _get_auth_headers(username="testuser", password="testpass123", role="user"):
    """Helper: register a user and return auth headers."""
    # Try to register; if already exists, login instead
    r = client.post("/auth/register", json={
        "username": username, "password": password
    })
    if r.status_code == 409:  # user already exists
        r = client.post("/auth/login", json={
            "username": username, "password": password
        })
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _get_admin_headers():
    """Helper: get auth headers for an admin user."""
    # Create admin directly via database for testing
    from app.database import get_user, create_user
    from app.auth import hash_password, create_token
    user = get_user("testadmin")
    if not user:
        user = create_user("testadmin", hash_password("adminpass123"), role="admin")
    token = create_token(user["username"], user["role"])
    return {"Authorization": f"Bearer {token}"}


# ── Auth tests ───────────────────────────────────────────────
def test_register():
    r = client.post("/auth/register", json={
        "username": "newuser_test", "password": "pass123456"
    })
    assert r.status_code == 200
    data = r.json()
    assert "token" in data
    assert data["role"] == "user"


def test_register_duplicate():
    client.post("/auth/register", json={
        "username": "dupuser", "password": "pass123456"
    })
    r = client.post("/auth/register", json={
        "username": "dupuser", "password": "pass123456"
    })
    assert r.status_code == 409


def test_login():
    client.post("/auth/register", json={
        "username": "loginuser", "password": "pass123456"
    })
    r = client.post("/auth/login", json={
        "username": "loginuser", "password": "pass123456"
    })
    assert r.status_code == 200
    assert "token" in r.json()


def test_login_wrong_password():
    client.post("/auth/register", json={
        "username": "wrongpwuser", "password": "pass123456"
    })
    r = client.post("/auth/login", json={
        "username": "wrongpwuser", "password": "wrongpassword"
    })
    assert r.status_code == 401


def test_me_endpoint():
    headers = _get_auth_headers("meuser", "pass123456")
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == "meuser"
    assert r.json()["role"] == "user"


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
    r = client.post("/ask",
        json={"question": "What is this project about?"},
        headers=headers)
    assert r.status_code == 200
    assert len(r.text) > 0


def test_ask_k_out_of_range():
    headers = _get_auth_headers("kuser", "pass123456")
    r = client.post("/ask",
        json={"question": "valid question here", "k": 99},
        headers=headers)
    assert r.status_code == 422


# ── Ingest (requires admin role) ─────────────────────────────
def test_ingest_requires_auth():
    r = client.post("/ingest", json={"source": "https://example.com"})
    assert r.status_code in (401, 403)


def test_ingest_denied_for_user_role():
    headers = _get_auth_headers("regularuser", "pass123456")
    r = client.post("/ingest",
        json={"source": "https://example.com"},
        headers=headers)
    assert r.status_code == 403


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