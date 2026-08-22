import ipaddress
import json
import logging
import pathlib
import socket
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from app.auth import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    get_current_user,
    login_user,
    logout_user,
    register_user,
    require_role,
)
from app.chain import ask_stream, get_llm_stats
from app.config import configure_logging, get_settings
from app.vectorstore import VectorStore, collection_count, get_cache_stats
from scripts.ingest import ingest as run_ingest

configure_logging()
log = logging.getLogger(__name__)
settings = get_settings()

# ── Metrics ──────────────────────────────────────────────────
# NOTE: these counters live in process memory. With more than one uvicorn worker
# or replica, /metrics reports whichever process happened to serve the request.
# For a real deployment scrape a Prometheus endpoint and aggregate there.
_metrics = defaultdict(int)
_latencies: deque[float] = deque(maxlen=1000)  # bounded: an unbounded list leaks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: best-effort pre-warm of the retriever.

    WHY wrap in try/except? Pre-warming is a latency optimisation, NOT a startup
    requirement. If it fails — constrained memory, a slow model download — we
    must still serve; the models load lazily on first use. An exception escaping
    the ASGI lifespan makes uvicorn exit(3) and the container crash-loop, taking
    the site down because an *optimisation* failed.
    """
    log.info("Loading retrieval models...")
    try:
        from app.retriever_instance import warm

        warm()
        log.info("Ready — all models pre-warmed.")
    except Exception as e:  # noqa: BLE001 — deliberately broad; startup must not die
        log.warning("Pre-warm failed (%s); models will load on first request.", e)
    yield
    log.info("Shutting down.")


def _client_key(request: Request) -> str:
    """Rate-limit bucket key: the real client IP.

    Behind a reverse proxy (Azure App Service, Render, any ingress) the socket
    peer is the *proxy*, so every user would share one bucket. X-Forwarded-For's
    first entry is the original client — but it is trivially spoofable, so it is
    only trusted when TRUST_PROXY_HEADERS says we really are behind a proxy.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_key, storage_uri=settings.rate_limit_storage_uri)
app = FastAPI(title="DevDocs AI", version="1.1.0", lifespan=lifespan)
app.state.limiter = limiter
# Without this handler, exceeding a @limiter.limit(...) raises an unhandled
# RateLimitExceeded → HTTP 500 instead of the correct 429 Too Many Requests.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ───────────────────────────────
@app.middleware("http")
async def log_and_track(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    _metrics["total_requests"] += 1
    _latencies.append(ms)
    if response.status_code >= 400:
        _metrics["errors"] += 1
    # Correlation ID: a user can quote this and you can find the exact request
    # across every log line.
    response.headers["X-Request-Id"] = rid
    return response


# ── Request models ───────────────────────────────────────────
class AskRequest(BaseModel):
    # Caps prompt size — a 1 MB question is an expensive LLM call, i.e. a
    # denial-of-wallet attack.
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(5, ge=1, le=10)


class IngestRequest(BaseModel):
    source: str = Field(..., description="GitHub URL, web URL, or local PDF path")


class DeleteSourceRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Exact source string to purge")


# ── SSRF guard for ingest sources ────────────────────────────
def validate_ingest_source(source: str) -> None:
    """Reject URLs that resolve to private / internal addresses.

    WHY? /ingest makes the server fetch (URL) or clone (git) a remote target.
    Without this an operator could point it at cloud metadata
    (169.254.169.254), loopback, or internal services — a classic SSRF.

    Layers: scheme allow-list → DNS resolution → is_global check on every
    resolved address (an attacker controls DNS, so checking the hostname string
    is useless; you must check the resolved IP).

    ⚠️  Residual risk: DNS is resolved again by requests/git when the fetch
    actually happens, leaving a rebinding window. `loaders.load_url` refuses to
    follow redirects for the same reason. Ingest is admin-only, so this is
    defence-in-depth rather than the only thing standing in the way.
    """
    if "://" not in source:
        return  # local path (e.g. /data/manual.pdf) — admin-only, skip URL checks

    parsed = urlparse(source)
    if parsed.scheme not in ("http", "https"):
        # Reject file://, ftp://, gopher://, etc. — only http(s) may be fetched.
        raise HTTPException(400, "Only http(s) URL sources are allowed")
    host = parsed.hostname
    if not host:
        raise HTTPException(400, "Invalid URL — missing host")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise HTTPException(400, f"Cannot resolve host: {host}")

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise HTTPException(
                400, f"Refusing to fetch private/internal address ({ip}) for host '{host}'"
            )


# ── SSE helpers ──────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event.

    WHY SSE instead of the old plain-text "\\n\\n||SOURCES||" sentinel? The
    sentinel was ambiguous (an LLM emitting that literal string would corrupt
    the split) and untyped. SSE is a standard framing with named events, and
    JSON-encoding the payload keeps newlines inside a token from breaking the
    protocol — `data:` lines cannot contain raw newlines.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Auth routes (public) ────────────────────────────────────
@app.post(
    "/auth/register",
    tags=["auth"],
    response_model=AuthResponse,
    summary="Register a new user account",
)
async def register(body: RegisterRequest):
    return register_user(body)


@app.post(
    "/auth/login",
    tags=["auth"],
    response_model=AuthResponse,
    summary="Login and receive a JWT token",
)
async def login(body: LoginRequest):
    return login_user(body)


@app.get("/auth/me", tags=["auth"], summary="Get current user info")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}


@app.post("/auth/logout", tags=["auth"], summary="Revoke the presented token")
async def logout(user: dict = Depends(get_current_user)):
    return logout_user(user)


# ── Query routes (user + admin) ──────────────────────────────
@app.post("/ask", tags=["query"], summary="Ask a question — streams answer tokens over SSE")
@limiter.limit(settings.ask_rate_limit)
async def ask(request: Request, body: AskRequest, user: dict = Depends(require_role("user"))):
    _metrics["ask_requests"] += 1

    async def generate():
        try:
            async for kind, payload in ask_stream(body.question, k=body.k):
                if kind == "token":
                    yield _sse("token", {"text": payload})
                elif kind == "sources":
                    yield _sse("sources", {"sources": payload})
        except Exception as e:
            # Once streaming starts the 200 is already committed, so errors can
            # only be delivered in-band — as a typed event the client can render
            # differently from answer text.
            _metrics["chain_errors"] += 1
            log.exception("Streaming chain failed")
            yield _sse("error", {"message": str(e)})
        finally:
            yield _sse("done", {})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx-style proxies buffering the stream
        },
    )


# ── Ingest (admin only) ──────────────────────────────────────
# Bounded job registry. Same per-process caveat as _metrics: with multiple
# workers a status poll may land on a worker that never saw the job. A shared
# store (Redis) or a real task queue is the fix at that point.
_JOBS_MAX = 100
_jobs: OrderedDict[str, dict] = OrderedDict()


def _run_ingest_job(job_id: str, source: str) -> None:
    """Execute an ingest and record the outcome. Runs in a worker thread."""
    job = _jobs.get(job_id)
    if job is None:  # evicted by a burst of newer jobs before this one started
        log.warning("Ingest job %s vanished from the registry before running", job_id)
        return
    job.update(status="running", started_at=datetime.now(UTC).isoformat())
    try:
        added = run_ingest(source)
        # The BM25 index is in-memory and was built at retriever construction;
        # without this rebuild the newly ingested chunks would be reachable by
        # dense search but invisible to sparse search until a process restart.
        from app.retriever_instance import get_retriever

        indexed = get_retriever().rebuild_bm25()
        job.update(
            status="succeeded",
            chunks_added=added,
            total_chunks=VectorStore().count(),
            bm25_documents=indexed,
        )
    except Exception as e:
        log.exception("Ingest job %s failed", job_id)
        job.update(status="failed", error=str(e))
    finally:
        job["finished_at"] = datetime.now(UTC).isoformat()


@app.post(
    "/ingest",
    tags=["ingest"],
    status_code=202,
    summary="Queue ingestion of a GitHub repo, URL, or PDF (user or admin)",
)
@limiter.limit(settings.ingest_rate_limit)
async def ingest_endpoint(
    request: Request,
    body: IngestRequest,
    background: BackgroundTasks,
    user: dict = Depends(require_role("user")),
):
    """Accepts the job and returns immediately.

    WHY 202 instead of doing the work inline? Cloning + chunking + embedding a
    real repository takes minutes, and load balancers cut idle connections at
    ~230s. The old synchronous handler produced a 504 while the work carried on
    invisibly, with no way to check progress.
    """
    validate_ingest_source(body.source)  # raises 400 before touching the network

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "source": body.source,
        "status": "queued",
        "queued_at": datetime.now(UTC).isoformat(),
    }
    while len(_jobs) > _JOBS_MAX:
        _jobs.popitem(last=False)

    # BackgroundTasks runs sync callables in a threadpool, so the blocking
    # clone/embed work never occupies the event loop.
    background.add_task(_run_ingest_job, job_id, body.source)
    return {"job_id": job_id, "status": "queued", "source": body.source}


@app.get("/ingest/{job_id}", tags=["ingest"], summary="Poll an ingest job (user or admin)")
async def ingest_status(job_id: str, user: dict = Depends(require_role("user"))):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id (it may have been evicted)")
    return job


@app.delete("/sources", tags=["ingest"], summary="Delete every chunk from a source (admin only)")
async def delete_source(body: DeleteSourceRequest, user: dict = Depends(require_role("admin"))):
    """Purge a source. Without this, upsert-only ingestion means a document
    deleted upstream stays retrievable forever."""
    deleted = await run_in_threadpool(VectorStore().delete_by_source, body.source)
    from app.retriever_instance import get_retriever

    await run_in_threadpool(get_retriever().rebuild_bm25)
    return {"status": "ok", "deleted_chunks": deleted, "source": body.source}


# ── Admin ────────────────────────────────────────────────────
@app.get("/metrics", tags=["admin"], summary="Operational metrics (admin only)")
async def metrics(user: dict = Depends(require_role("admin"))):
    lats = sorted(_latencies)
    # p95, not the mean: averages hide tail latency, and the tail is what users
    # actually complain about.
    p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)] if lats else 0
    return {
        **dict(_metrics),
        "p95_latency_ms": round(p95, 1),
        "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
        **get_cache_stats(),
        **get_llm_stats(),
    }


@app.get("/admin/users", tags=["admin"], summary="List all registered users (admin only)")
async def list_all_users(user: dict = Depends(require_role("admin"))):
    from app.database import list_users

    return {"users": list_users()}


# ── Public ops routes ────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    # collection_count() deliberately avoids constructing a VectorStore, which
    # would load the embedding model on a probe that runs every 30 seconds.
    return {"status": "ok", "chunks_in_db": collection_count()}


# ── Serve frontend static files ──────────────────────────────
# IMPORTANT: must be LAST — mounts match in registration order and this one
# catches every unmatched path.
_frontend_dir = pathlib.Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
