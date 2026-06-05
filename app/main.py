import os, time, uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.chain import ask_stream
from app.vectorstore import VectorStore
from scripts.ingest import ingest as run_ingest
from dotenv import load_dotenv
load_dotenv()

_metrics = defaultdict(int)
_latencies: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading retrieval models...")
    from app.hybrid_retriever import HybridRetriever
    app.state.retriever = HybridRetriever()  # loads CrossEncoder + BM25
    print("[startup] Ready.")
    yield
    print("[shutdown] Cleaning up.")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="DevDocs AI", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
security = HTTPBearer()

def verify_key(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != os.getenv("API_KEY", "dev-key"):
        raise HTTPException(401, "Invalid API key")
    return creds.credentials


# ── Request logging middleware ───────────────────────────
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


# ── Models ───────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int        = Field(5, ge=1, le=10)

class IngestRequest(BaseModel):
    source: str = Field(..., description="GitHub URL, web URL, or local PDF path")



# ── Routes ───────────────────────────────────────────────
@app.post("/ask", tags=["query"],
         summary="Ask a question — streams answer tokens")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest, _=Depends(verify_key)):
    _metrics["ask_requests"] += 1
    async def generate():
        try:
            async for token in ask_stream(body.question, k=body.k):
                yield token
        except Exception as e:
            _metrics["chain_errors"] += 1
            yield f"\n\n[Error: {str(e)}]"
    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/ingest", tags=["admin"],
          summary="Ingest a GitHub repo, URL, or PDF")
@limiter.limit("5/minute")
async def ingest_endpoint(request: Request, body: IngestRequest,
                          _=Depends(verify_key)):
    try:
        # WHY run_ingest in a thread executor?
        # run_ingest does file I/O + embedding computation — both
        # blocking operations. Running in a thread pool prevents
        # them from blocking the async event loop.
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_ingest, body.source)
        vs = VectorStore()
        return {"status": "ok", "total_chunks": vs.count(),
                "source": body.source}
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")
    
@app.get("/metrics", tags=["ops"])
def metrics():
    p95 = sorted(_latencies)[int(len(_latencies)*0.95)] if _latencies else 0
    return {**dict(_metrics),
            "p95_latency_ms": round(p95, 1),
            "avg_latency_ms": round(sum(_latencies)/len(_latencies),1) if _latencies else 0}

@app.get("/health", tags=["ops"])
def health(): return {"status": "ok", "chunks_in_db": VectorStore().count()}