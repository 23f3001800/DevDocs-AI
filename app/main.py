import pathlib
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    get_current_user,
    login_user,
    register_user,
    require_role,
)
from app.chain import ask_stream, get_llm_stats
from app.vectorstore import VectorStore, get_cache_stats
from scripts.ingest import ingest as run_ingest

load_dotenv()

# ── Metrics (bounded deque prevents memory leak) ─────────────
_metrics = defaultdict(int)
_latencies: deque[float] = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-warm retriever for zero cold-start latency."""
    print("[startup] Loading retrieval models...")
    from app.retriever_instance import warm

    warm()  # forces model load + dummy query
    print("[startup] Ready — all models pre-warmed.")
    yield
    print("[shutdown] Cleaning up.")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="DevDocs AI", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    response.headers["X-Request-Id"] = rid
    return response


# ── Request models ───────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(5, ge=1, le=10)


class IngestRequest(BaseModel):
    source: str = Field(..., description="GitHub URL, web URL, or local PDF path")


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


# ── Query routes (user + admin) ──────────────────────────────
@app.post("/ask", tags=["query"], summary="Ask a question — streams answer tokens")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest, user: dict = Depends(require_role("user"))):
    _metrics["ask_requests"] += 1

    async def generate():
        try:
            async for token in ask_stream(body.question, k=body.k):
                yield token
        except Exception as e:
            _metrics["chain_errors"] += 1
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


# ── Ingest route (user + admin) ──────────────────────────────
@app.post("/ingest", tags=["ingest"], summary="Ingest a GitHub repo, URL, or PDF")
@limiter.limit("5/minute")
async def ingest_endpoint(
    request: Request, body: IngestRequest, user: dict = Depends(require_role("user"))
):
    try:
        # WHY run_ingest in a thread executor?
        # run_ingest does file I/O + embedding computation — both
        # blocking operations. Running in a thread pool prevents
        # them from blocking the async event loop.
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_ingest, body.source)
        vs = VectorStore()
        return {"status": "ok", "total_chunks": vs.count(), "source": body.source}
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@app.get("/metrics", tags=["admin"], summary="Operational metrics (admin only)")
async def metrics(user: dict = Depends(require_role("admin"))):
    lats = list(_latencies)
    p95 = sorted(lats)[int(len(lats) * 0.95)] if lats else 0
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
    return {"status": "ok", "chunks_in_db": VectorStore().count()}


# ── Serve frontend static files ──────────────────────────────
# IMPORTANT: This must be the LAST route — it catches all unmatched paths.
# Mount at "/" so index.html is served at the root URL.
_frontend_dir = pathlib.Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
