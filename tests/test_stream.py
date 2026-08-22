"""
Streaming path: SSE framing and the mid-stream fallback contract.

This file used to be a manual script that called the live API. It is now real
pytest coverage that runs offline against the MockProvider and fakes.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.llm_providers import LLMManager, LLMResponse
from app.main import _sse, app
from tests.conftest import session_headers

client = TestClient(app)


# ── SSE framing ──────────────────────────────────────────────
def test_sse_frame_shape():
    frame = _sse("token", {"text": "hello"})
    assert frame.startswith("event: token\n")
    assert frame.endswith("\n\n")
    assert json.loads(frame.split("data: ", 1)[1].strip()) == {"text": "hello"}


def test_sse_survives_newlines_in_a_token():
    """`data:` lines cannot contain raw newlines — JSON encoding is what makes
    a markdown code block safe to stream."""
    frame = _sse("token", {"text": "line1\n\nline2"})
    body = frame.split("data: ", 1)[1]
    assert body.count("\n") == 2  # only the two framing newlines
    assert json.loads(body.strip())["text"] == "line1\n\nline2"


def test_ask_streams_sse_events():
    r = client.post(
        "/ask", json={"question": "What is this project about?"}, headers=session_headers()
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in r.text
    # Either real answer tokens or the "nothing ingested" notice, but a sources
    # or error event must always arrive before done.
    assert "event: sources" in r.text or "event: error" in r.text


def test_ask_no_longer_uses_the_sources_sentinel():
    r = client.post("/ask", json={"question": "anything at all here"}, headers=session_headers())
    assert "||SOURCES||" not in r.text


# ── Provider fallback during streaming ───────────────────────
def _provider(name, *, fail_at=None):
    """fail_at=None → never fails; 0 → fails before any token; 2 → mid-stream."""

    class _P:
        def __init__(self):
            self.name = name
            self.model = f"{name}-model"

        def generate(self, system, user_message, max_tokens=1024):
            return LLMResponse(text=f"ok-{name}", provider=name, model=self.model)

        async def stream(self, system, user_message, max_tokens=1024):
            for i in range(4):
                if fail_at is not None and i == fail_at:
                    raise RuntimeError(f"{name} exploded at {i}")
                yield f"{name}{i} "

    return _P()


def _manager(providers):
    mgr = LLMManager.__new__(LLMManager)  # bypass env detection
    mgr.providers = providers
    return mgr


async def test_stream_falls_back_when_primary_fails_before_any_token():
    mgr = _manager([_provider("primary", fail_at=0), _provider("backup")])
    out = "".join([t async for t in mgr.stream("s", "u")])
    assert out.strip() == "backup0 backup1 backup2 backup3"
    assert "primary" not in out


async def test_stream_does_not_duplicate_output_on_mid_stream_failure():
    """The bug: falling through to the next provider after tokens had already
    reached the client concatenated a partial answer with a complete one, so the
    user saw the same question answered twice in a single message."""
    mgr = _manager([_provider("primary", fail_at=2), _provider("backup")])

    collected = []
    with pytest.raises(RuntimeError, match="Stream interrupted"):
        async for token in mgr.stream("s", "u"):
            collected.append(token)

    out = "".join(collected)
    assert out.strip() == "primary0 primary1"
    assert "backup" not in out  # never restarted from scratch


async def test_stream_raises_when_every_provider_fails_up_front():
    mgr = _manager([_provider("a", fail_at=0), _provider("b", fail_at=0)])
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        async for _ in mgr.stream("s", "u"):
            pass
