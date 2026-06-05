import pytest
from fastapi.testclient import TestClient
from app.main import app

# WHY TestClient instead of a real running server?
# TestClient runs the app in-process — no network overhead,
# deterministic, fast, and works in CI without port conflicts.
client = TestClient(app)
HEADERS = {"Authorization": "Bearer dev-secret-key-change-in-prod"}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "chunks_in_db" in r.json()

def test_ask_requires_auth():
    # No auth header → must get 403, never 200
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code in (401, 403)

def test_ask_wrong_key():
    r = client.post("/ask",
        json={"question": "hello"},
        headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401

def test_ask_question_too_short():
    # min_length=3 in AskRequest should reject 1-char queries
    r = client.post("/ask", json={"question": "hi"}, headers=HEADERS)
    assert r.status_code == 422  # Pydantic validation error


def test_ask_returns_streaming_response():
    # With valid auth + question → response body must not be empty
    # We don't validate the content (depends on ingested docs)
    # but we verify the endpoint responds successfully
    r = client.post("/ask",
        json={"question": "What is this project about?"},
        headers=HEADERS)
    assert r.status_code == 200
    assert len(r.text) > 0

def test_ingest_requires_auth():
    r = client.post("/ingest", json={"source": "https://example.com"})
    assert r.status_code in (401, 403)

def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "p95_latency_ms" in data

def test_ask_k_out_of_range():
    # k > 10 should fail Pydantic validation
    r = client.post("/ask",
        json={"question": "valid question here", "k": 99},
        headers=HEADERS)
    assert r.status_code == 422