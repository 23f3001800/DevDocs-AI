import hmac
import json
import logging
import pathlib
import secrets
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from app.chain import ask_stream, get_llm_stats
from app.config import configure_logging, get_settings
from app.database import (
    add_message,
    create_conversation,
    delete_conversation,
    delete_source_row,
    get_conversation,
    get_conversation_messages,
    get_usage_count,
    increment_usage,
    list_conversations,
    list_sources_for_user,
    record_source,
)
from app.security import SSRFError, resolve_public_ips
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
_ask_latencies: deque[float] = deque(maxlen=1000)
_ttfts: deque[float] = deque(maxlen=1000)

# Owners are server-issued, signed HttpOnly cookie values. A client cannot
# impersonate another session by supplying a UUID in a request header.
_SESSION_COOKIE_NAME = "devdocs_session"
_SESSION_TOKEN_VERSION = "v1"
_EPHEMERAL_SESSION_SECRET = secrets.token_bytes(32)


def _session_secret() -> bytes:
    configured = settings.session_secret.strip()
    return configured.encode("utf-8") if configured else _EPHEMERAL_SESSION_SECRET


def make_session_token(session_id: str | None = None, issued_at: int | None = None) -> str:
    """Create a tamper-evident anonymous-session token for a cookie."""
    owner = str(uuid.UUID(session_id)) if session_id else str(uuid.uuid4())
    timestamp = int(time.time()) if issued_at is None else int(issued_at)
    payload = f"{_SESSION_TOKEN_VERSION}.{owner}.{timestamp}"
    signature = hmac.new(_session_secret(), payload.encode("utf-8"), "sha256").hexdigest()
    return f"{payload}.{signature}"


def session_id_from_token(token: str | None) -> str | None:
    """Return the cookie owner only when its signature and expiry are valid."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    version, owner, raw_timestamp, signature = parts
    if version != _SESSION_TOKEN_VERSION:
        return None
    try:
        canonical_owner = str(uuid.UUID(owner))
        issued_at = int(raw_timestamp)
    except (ValueError, TypeError):
        return None
    now = int(time.time())
    if issued_at > now + 60 or now - issued_at > settings.session_max_age_seconds:
        return None
    payload = f"{version}.{canonical_owner}.{issued_at}"
    expected = hmac.new(_session_secret(), payload.encode("utf-8"), "sha256").hexdigest()
    return canonical_owner if hmac.compare_digest(signature, expected) else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: best-effort pre-warm of the retriever.

    Wrapped in try/except because pre-warming is only a latency optimisation
    (models load lazily on first use); letting it escape the lifespan would
    crash-loop the container over a failed optimisation.
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
    session_id = session_id_from_token(request.cookies.get(_SESSION_COOKIE_NAME))
    new_session_token = None
    if session_id is None:
        session_id = str(uuid.uuid4())
        new_session_token = make_session_token(session_id)
    request.state.session_id = session_id
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    _metrics["total_requests"] += 1
    _latencies.append(ms)
    if response.status_code >= 400:
        _metrics["errors"] += 1
    # Correlation ID: a user can quote this and you can find the exact request
    # across every log line.
    response.headers["X-Request-Id"] = rid
    if new_session_token:
        response.set_cookie(
            key=_SESSION_COOKIE_NAME,
            value=new_session_token,
            max_age=settings.session_max_age_seconds,
            httponly=True,
            secure=settings.session_cookie_secure or settings.is_production,
            samesite="lax",
            path="/",
        )
    vary = response.headers.get("Vary")
    response.headers["Vary"] = f"{vary}, Cookie" if vary else "Cookie"
    return response


# ── Request models ───────────────────────────────────────────
class AskRequest(BaseModel):
    # Caps prompt size — a 1 MB question is an expensive LLM call, i.e. a
    # denial-of-wallet attack.
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(5, ge=1, le=10)
    conversation_id: int | None = Field(
        None, description="Append to an existing conversation; omit to start a new one"
    )


class IngestRequest(BaseModel):
    source: str = Field(..., description="GitHub URL, web URL, or local PDF path")


class DeleteSourceRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Exact source string to purge")


class SearchSourcesRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(None, max_length=200)


# ── SSRF guard for ingest sources ────────────────────────────
def validate_ingest_source(source: str) -> None:
    """Reject URLs that resolve to private / internal addresses.

    WHY? /ingest makes the server fetch (URL) or clone (git) a remote target.
    Without this an operator could point it at cloud metadata
    (169.254.169.254), loopback, or internal services — a classic SSRF.

    Layers: scheme allow-list → DNS resolution → is_global check on every
    resolved address (an attacker controls DNS, so checking the hostname string
    is useless; you must check the resolved IP).

    ⚠️  DNS is resolved again by requests/git when the fetch actually happens
    (a rebinding window), so `app.loaders` re-runs this same check immediately
    before the network hop. This submission-time check exists to fail fast and
    return a clean 400 instead of a job that silently fails minutes later.
    `loaders.load_url` also refuses to follow redirects, for the same reason.
    """
    if "://" not in source:
        return  # local path (e.g. /data/manual.pdf) — no network fetch, skip URL checks

    parsed = urlparse(source)
    if parsed.scheme not in ("http", "https"):
        # Reject file://, ftp://, gopher://, etc. — only http(s) may be fetched.
        raise HTTPException(400, "Only http(s) URL sources are allowed")
    host = parsed.hostname
    if not host:
        raise HTTPException(400, "Invalid URL — missing host")

    try:
        resolve_public_ips(host)
    except SSRFError as e:
        raise HTTPException(400, str(e))


# ── SSE helpers ──────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event.

    SSE gives typed, named events; JSON-encoding the payload keeps newlines
    inside a token from breaking the protocol (`data:` lines can't contain them).
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Anonymous session identity ───────────────────────────────
# The middleware verifies a signed, HttpOnly cookie before choosing the owner.
# Client-supplied X-Session-Id values are ignored and cannot authorize access.
_API_KEY_HEADER = "X-Api-Key"


def get_session_id(request: Request) -> str:
    """Read the verified owner that the session middleware attached."""
    session_id = getattr(request.state, "session_id", None)
    if not isinstance(session_id, str):
        raise HTTPException(401, "A valid session cookie is required")
    return session_id


def get_api_key(request: Request) -> str | None:
    """Optional BYOK header — the caller's own Gemini API key for this request."""
    return request.headers.get(_API_KEY_HEADER, "").strip() or None


def _is_quota_error(exc: Exception) -> bool:
    """Best-effort detection of a Gemini rate-limit/quota failure by message,
    since the SDK's error hierarchy varies by transport/version."""
    text = str(exc).lower()
    return "429" in text or "quota" in text or "resource_exhausted" in text


# ── Query routes ──────────────────────────────────────────────
@app.post("/ask", tags=["query"], summary="Ask a question — streams answer tokens over SSE")
@limiter.limit(settings.ask_rate_limit)
async def ask(
    request: Request,
    body: AskRequest,
    session_id: str = Depends(get_session_id),
    api_key: str | None = Depends(get_api_key),
):
    _metrics["ask_requests"] += 1
    ask_started = time.perf_counter()
    owner_filter = session_id  # retrieval is per-session private

    # Free daily limit only applies to the server's own key; checked before
    # touching conversation/message tables so a blocked question leaves no
    # half-written turn in history.
    if not api_key:
        used = await run_in_threadpool(get_usage_count, session_id)
        if used >= settings.free_daily_limit:

            async def limit_reached():
                yield _sse(
                    "error",
                    {
                        "code": "limit_reached",
                        "message": (
                            f"You've used your {settings.free_daily_limit} free questions "
                            "today. Add your own Gemini API key to keep asking."
                        ),
                    },
                )
                yield _sse("done", {})

            return StreamingResponse(
                limit_reached(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        await run_in_threadpool(increment_usage, session_id)

    # Resolve/create the conversation before streaming starts, so a bad
    # conversation_id fails fast as a normal HTTP error, not an in-band one.
    conversation_id = body.conversation_id
    is_new_conversation = False
    if conversation_id is not None:
        conv = await run_in_threadpool(get_conversation, conversation_id)
        # 404, not 403 — avoids confirming another session's id exists.
        if not conv or conv["owner"] != session_id:
            raise HTTPException(404, "Conversation not found")
    else:
        conv = await run_in_threadpool(create_conversation, session_id, body.question[:60])
        conversation_id = conv["id"]
        is_new_conversation = True

    await run_in_threadpool(add_message, conversation_id, "user", body.question, None)

    async def generate():
        collected: list[str] = []
        collected_sources: list[str] = []
        failed = False
        emitted_first_token = False
        counted_citation = False
        try:
            if is_new_conversation:
                # Emitted first so the client can track this conversation
                # before any answer token arrives.
                yield _sse("meta", {"conversation_id": conversation_id})
            async for kind, payload in ask_stream(
                body.question, k=body.k, owner=owner_filter, api_key=api_key
            ):
                if kind == "token":
                    if not emitted_first_token:
                        emitted_first_token = True
                        _ttfts.append((time.perf_counter() - ask_started) * 1000)
                    collected.append(payload)
                    yield _sse("token", {"text": payload})
                elif kind == "sources":
                    collected_sources = payload
                    if payload and not counted_citation:
                        _metrics["answers_with_citations"] += 1
                        counted_citation = True
                    yield _sse("sources", {"sources": payload})
        except Exception as e:
            # 200 is already committed once streaming starts — errors go
            # in-band as a typed event instead.
            failed = True
            _metrics["chain_errors"] += 1
            log.exception("Streaming chain failed")
            if _is_quota_error(e):
                yield _sse(
                    "error",
                    {
                        "code": "quota",
                        "message": (
                            "The Gemini API key in use has hit its quota. "
                            "Add your own API key to keep asking."
                        ),
                    },
                )
            else:
                # Generic message — str(e) can carry provider internals the
                # client has no business seeing; specifics go to the log.
                yield _sse(
                    "error",
                    {
                        "code": "error",
                        "message": "Something went wrong answering your question. Please try again.",
                    },
                )
        finally:
            _ask_latencies.append((time.perf_counter() - ask_started) * 1000)
            # Persist only on success — question was already saved above.
            if not failed and collected:
                answer_text = "".join(collected)
                try:
                    await run_in_threadpool(
                        add_message, conversation_id, "assistant", answer_text, collected_sources
                    )
                except Exception:
                    log.exception(
                        "Failed to persist assistant message for conversation %s", conversation_id
                    )
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


@app.get("/usage", tags=["query"], summary="Today's free-question usage for this session")
async def usage(session_id: str = Depends(get_session_id)):
    used = await run_in_threadpool(get_usage_count, session_id)
    limit = settings.free_daily_limit
    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


# ── Ingest job registry ───────────────────────────────────────
# Bounded job registry. Same per-process caveat as _metrics: with multiple
# workers a status poll may land on a worker that never saw the job. A shared
# store (Redis) or a real task queue is the fix at that point.
_JOBS_MAX = 100
_jobs: OrderedDict[str, dict] = OrderedDict()

_TERMINAL_STATUSES = ("succeeded", "failed")


def _evict_completed_jobs() -> None:
    """Trim the job store back to _JOBS_MAX, oldest-first, but only among
    terminal jobs. Plain FIFO could evict a still-queued/running job mid-flight,
    abandoning live work; only succeeded/failed jobs are safe to drop.
    """
    while len(_jobs) > _JOBS_MAX:
        for jid, j in _jobs.items():
            if j.get("status") in _TERMINAL_STATUSES:
                del _jobs[jid]
                break
        else:
            # Nothing evictable — every remaining job is still in flight.
            # Better to run briefly over the soft cap than drop live work.
            break


def _run_ingest_job(job_id: str, source: str, owner: str) -> None:
    """Execute an ingest and record the outcome. Runs in a worker thread."""
    job = _jobs.get(job_id)
    if job is None:  # evicted by a burst of newer jobs before this one started
        log.warning("Ingest job %s vanished from the registry before running", job_id)
        return
    job.update(status="running", started_at=datetime.now(UTC).isoformat())
    try:
        result = run_ingest(source, owner=owner)
        added = result["chunks_added"]
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
        if added:
            record_source(owner, source, result["kind"], added, result["commit_sha"])
    except Exception as e:
        log.exception("Ingest job %s failed", job_id)
        job.update(status="failed", error=str(e))
    finally:
        job["finished_at"] = datetime.now(UTC).isoformat()


@app.post(
    "/ingest",
    tags=["ingest"],
    status_code=202,
    summary="Queue ingestion of a GitHub repo, URL, or PDF",
)
@limiter.limit(settings.ingest_rate_limit)
async def ingest_endpoint(
    request: Request,
    body: IngestRequest,
    background: BackgroundTasks,
    session_id: str = Depends(get_session_id),
):
    """Accepts the job and returns 202 immediately — cloning + chunking +
    embedding a real repo takes minutes, past the ~230s load-balancer idle cut.
    """
    # socket.getaddrinfo is a blocking syscall; run it off the event loop so a
    # slow or hung DNS server can't stall every other request being served by
    # this worker.
    await run_in_threadpool(validate_ingest_source, body.source)  # raises 400 before the network

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "source": body.source,
        "status": "queued",
        "queued_at": datetime.now(UTC).isoformat(),
        "submitted_by": session_id,
    }
    _evict_completed_jobs()

    # BackgroundTasks runs sync callables in a threadpool, so the blocking
    # clone/embed work never occupies the event loop.
    background.add_task(_run_ingest_job, job_id, body.source, session_id)
    return {"job_id": job_id, "status": "queued", "source": body.source}


# ── PDF upload ingest ─────────────────────────────────────────
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # ~20MB


@app.post(
    "/ingest/upload",
    tags=["ingest"],
    status_code=202,
    summary="Upload a PDF and queue it for ingestion",
)
@limiter.limit(settings.ingest_rate_limit)
async def ingest_upload(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
):
    """Accepts a PDF upload only. No SSRF check applies here — the "fetch" is
    a local file save, not a request to an attacker-controlled URL (unlike
    /ingest, which does resolve and fetch whatever source string it is given).
    """
    filename = file.filename or ""
    ext = pathlib.Path(filename).suffix.lower()
    if ext != ".pdf" or file.content_type != "application/pdf":
        raise HTTPException(415, "Only PDF uploads are accepted (.pdf, application/pdf)")

    # Read one byte past the cap so an oversized file can be rejected without
    # trusting a (spoofable) Content-Length header.
    data = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large — max 20MB")
    if not data:
        raise HTTPException(400, "Empty file")

    upload_dir = pathlib.Path(settings.upload_dir)
    await run_in_threadpool(upload_dir.mkdir, parents=True, exist_ok=True)
    # A random prefix on top of the sanitised original name — Path(...).name
    # strips any directory components, closing a path-traversal upload
    # (e.g. "../../etc/passwd") — and avoids two uploads of "doc.pdf" from
    # different sessions colliding on disk.
    safe_name = f"{uuid.uuid4().hex[:12]}_{pathlib.Path(filename).name}"
    dest = upload_dir / safe_name
    await run_in_threadpool(dest.write_bytes, data)

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "source": str(dest),
        "status": "queued",
        "queued_at": datetime.now(UTC).isoformat(),
        "submitted_by": session_id,
    }
    _evict_completed_jobs()
    background.add_task(_run_ingest_job, job_id, str(dest), session_id)
    return {"job_id": job_id, "status": "queued", "source": str(dest)}


@app.get("/ingest/{job_id}", tags=["ingest"], summary="Poll an ingest job (submitter only)")
async def ingest_status(job_id: str, session_id: str = Depends(get_session_id)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id (it may have been evicted)")
    # Scoped to the submitting session — job ids are short and guessable.
    if job.get("submitted_by") != session_id:
        raise HTTPException(403, "You may only view ingest jobs you submitted")
    return job


@app.get("/sources/mine", tags=["ingest"], summary="List the caller's ingested sources")
async def sources_mine(session_id: str = Depends(get_session_id)):
    sources = await run_in_threadpool(list_sources_for_user, session_id)
    return {"sources": sources}


@app.delete(
    "/sources",
    tags=["ingest"],
    summary="Delete every chunk from a source you own",
)
async def delete_source(body: DeleteSourceRequest, session_id: str = Depends(get_session_id)):
    """Purge a source THIS session ingested. Both the `sources` row check and
    the vector-store delete are scoped to session_id, so it can never touch
    another session's copy of the same source string."""
    mine = await run_in_threadpool(list_sources_for_user, session_id)
    if not any(s["source"] == body.source for s in mine):
        raise HTTPException(403, "You may only delete sources you own")

    deleted = await run_in_threadpool(VectorStore().delete_by_source, body.source, session_id)
    from app.retriever_instance import get_retriever

    await run_in_threadpool(get_retriever().rebuild_bm25)
    await run_in_threadpool(delete_source_row, body.source, session_id)
    return {"status": "ok", "deleted_chunks": deleted, "source": body.source}


# ── Web search — candidate sources to ingest ─────────────────
def _resolve_ddg_href(href: str) -> str:
    """DuckDuckGo's HTML result links are often a redirector
    (//duckduckgo.com/l/?uddg=<url-encoded target>) rather than the target
    URL itself. Unwrap it so the caller gets a URL they can actually ingest.
    """
    if "uddg=" not in href:
        return href
    parsed = urlparse(href if href.startswith("http") else f"https:{href}")
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else href


def _duckduckgo_search(query: str) -> dict:
    """Best-effort web search via DuckDuckGo's no-JS HTML endpoint. Never raises:
    it's a convenience for picking a link to /ingest, so failures degrade to an
    empty result list rather than a 500.
    """
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=get_settings().ingest_timeout_seconds,
            headers={"User-Agent": "DevDocsAI/1.0"},
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for a in soup.select("a.result__a")[:8]:
            title = a.get_text(strip=True)
            href = a.get("href")
            if title and href:
                results.append({"title": title, "url": _resolve_ddg_href(href)})
        if not results:
            return {"results": [], "message": "No results found"}
        return {"results": results}
    except Exception as e:
        log.warning("search-sources failed for %r: %s", query, e)
        return {"results": [], "message": "Search failed — try again, or paste a URL directly."}


@app.post(
    "/search-sources",
    tags=["ingest"],
    summary="Web search for candidate sources to ingest",
)
@limiter.limit(settings.ingest_rate_limit)
async def search_sources(
    request: Request, body: SearchSourcesRequest, session_id: str = Depends(get_session_id)
):
    return await run_in_threadpool(_duckduckgo_search, body.query)


# ── Chat history (per-session private) ───────────────────────
@app.get("/conversations", tags=["chat"], summary="List the caller's conversations")
async def list_conversations_endpoint(session_id: str = Depends(get_session_id)):
    conversations = await run_in_threadpool(list_conversations, session_id)
    return {"conversations": conversations}


@app.post("/conversations", tags=["chat"], summary="Create a new (empty) conversation")
async def create_conversation_endpoint(
    body: ConversationCreateRequest, session_id: str = Depends(get_session_id)
):
    conv = await run_in_threadpool(create_conversation, session_id, body.title)
    return {"id": conv["id"]}


@app.get(
    "/conversations/{conversation_id}",
    tags=["chat"],
    summary="Get a conversation and its messages (owner only)",
)
async def get_conversation_endpoint(
    conversation_id: int, session_id: str = Depends(get_session_id)
):
    conv = await run_in_threadpool(get_conversation, conversation_id)
    # 404 (not 403) for someone else's conversation — a 403 would confirm the
    # id exists, which is itself information a stranger shouldn't get.
    if not conv or conv["owner"] != session_id:
        raise HTTPException(404, "Conversation not found")
    messages = await run_in_threadpool(get_conversation_messages, conversation_id)
    return {**conv, "messages": messages}


@app.delete(
    "/conversations/{conversation_id}",
    tags=["chat"],
    summary="Delete a conversation (owner only)",
)
async def delete_conversation_endpoint(
    conversation_id: int, session_id: str = Depends(get_session_id)
):
    conv = await run_in_threadpool(get_conversation, conversation_id)
    if not conv or conv["owner"] != session_id:
        raise HTTPException(404, "Conversation not found")
    await run_in_threadpool(delete_conversation, conversation_id)
    return {"status": "ok"}


# ── Operational metrics (public — no accounts left to gate it) ──
@app.get("/metrics", tags=["ops"], summary="Operational metrics")
async def metrics():
    lats = sorted(_ask_latencies or _latencies)
    ttfts = sorted(_ttfts)
    # p95, not the mean: averages hide tail latency, and the tail is what users
    # actually complain about.
    p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)] if lats else 0
    p95_ttft = ttfts[min(int(len(ttfts) * 0.95), len(ttfts) - 1)] if ttfts else 0
    return {
        **dict(_metrics),
        "p95_latency_ms": round(p95, 1),
        "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
        "p95_ttft_ms": round(p95_ttft, 1),
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0,
        "latency_samples": len(lats),
        "ttft_samples": len(ttfts),
        "latency_scope": "end_to_end_answer_stream" if _ask_latencies else "request_setup",
        **get_cache_stats(),
        **get_llm_stats(),
    }


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
