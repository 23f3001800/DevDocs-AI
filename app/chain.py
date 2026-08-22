import json
import logging
import math
import os
import time

from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.llm_providers import llm_manager
from app.models import RAGResponse
from app.retriever_instance import get_retriever

log = logging.getLogger(__name__)

# ── LangSmith tracing (optional) ─────────────────────────────
# Opt-in: when disabled, @traceable becomes a no-op so it costs nothing.
_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def _noop_traceable(*args, **kwargs):
    def decorator(fn):
        return fn

    # Handles both @traceable and @traceable(name=...) forms.
    return decorator if not args or not callable(args[0]) else args[0]


if _tracing_enabled:
    try:
        from langsmith import traceable
    except ImportError:
        traceable = _noop_traceable
else:
    traceable = _noop_traceable


# ── Metrics tracking ─────────────────────────────────────────
_llm_stats = {"calls": 0, "errors": 0, "total_ms": 0.0, "providers_used": {}}


def get_llm_stats() -> dict:
    """Return LLM call stats for /metrics."""
    calls = _llm_stats["calls"]
    return {
        "llm_calls": calls,
        "llm_errors": _llm_stats["errors"],
        "llm_avg_ms": round(_llm_stats["total_ms"] / calls, 1) if calls else 0,
        "providers_available": llm_manager.provider_names(),
        "providers_used": dict(_llm_stats["providers_used"]),
    }


SYSTEM = """You are DevDocs AI — a technical assistant that answers questions
about codebases and documentation.

Rules:
1. Answer ONLY from the provided context. Never invent information.
2. If the context doesn't contain the answer, set has_answer=false.
3. Include the file_path from context metadata in sources.
4. Estimate confidence: how well does the context support your answer?

Respond ONLY with valid JSON matching this exact schema — no preamble, no markdown:
{
  "answer": "string",
  "sources": ["file_path_1", "file_path_2"],
  "confidence": 0.0-1.0,
  "has_answer": true/false
}"""


# Separate streaming prompt: partial JSON is unrenderable, so the streaming path
# emits prose and sends sources out-of-band as a separate SSE event.
STREAM_SYSTEM = """You are DevDocs AI — a technical assistant that answers questions
about codebases and documentation.

Rules:
1. Answer ONLY from the provided context. Never invent information.
2. If the context does not contain the answer, say so plainly.
3. Write a clear, well-formatted Markdown answer. Do NOT wrap it in JSON.
4. Reference relevant file paths inline when they help the reader."""


def _relevance(rerank_score: float) -> float:
    """Sigmoid-squash a raw CrossEncoder logit (unbounded) into a 0..1 relevance
    the model can read like a probability when calibrating its confidence."""
    return 1.0 / (1.0 + math.exp(-rerank_score))


def _build_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """Format retrieved chunks into a context block, and collect sources.

    file_path is included in-band because the model can only cite what it can
    see; relevance is included so it can calibrate `confidence` — if every
    chunk scored low, retrieval was weak and the answer should be hedged.
    """
    parts, sources = [], []
    for i, chunk in enumerate(chunks):
        fp = chunk["metadata"].get("file_path", "unknown")
        sources.append(fp)
        relevance = _relevance(chunk["rerank_score"])
        parts.append(f"[Chunk {i + 1} | {fp} | relevance: {relevance:.2f}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts), sources


def _no_documents_message(owner: str | None) -> str:
    """A more actionable message when this session just hasn't ingested
    anything yet, vs. the no-owner case (offline scripts only) meaning the
    whole corpus is empty."""
    if owner:
        return "You haven't ingested any documents yet. Ingest a source first, then ask again."
    return "No documents have been ingested yet. Run scripts/ingest.py first."


@traceable(name="ask_sync", run_type="chain")
def ask(question: str, k: int = 5, owner: str | None = None, api_key: str | None = None) -> RAGResponse:
    """Synchronous RAG query — returns a structured RAGResponse.

    `owner` scopes retrieval to that session's own ingested chunks (`None`
    is offline-scripts-only). `api_key` (BYOK) uses that Gemini key instead
    of the server's own.
    """
    chunks = get_retriever().retrieve(question, k=k, owner=owner)

    if not chunks:
        return RAGResponse(
            answer=_no_documents_message(owner),
            sources=[],
            confidence=0.0,
            has_answer=False,
        )

    context, sources = _build_context(chunks)
    # dict.fromkeys dedupes while preserving retrieval order (set() would not).
    real_sources = list(dict.fromkeys(sources))
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    max_tokens = get_settings().llm_max_tokens

    t0 = time.time()
    _llm_stats["calls"] += 1
    try:
        result = llm_manager.generate(SYSTEM, user_message, max_tokens=max_tokens, api_key=api_key)
        _llm_stats["providers_used"][result.provider] = (
            _llm_stats["providers_used"].get(result.provider, 0) + 1
        )
    except Exception:
        _llm_stats["errors"] += 1
        raise
    finally:
        _llm_stats["total_ms"] += (time.time() - t0) * 1000

    raw = result.text

    # Parse with self-repair: feed a parse error back to the model to fix a
    # trailing comma / markdown fence; the final fallback keeps the endpoint
    # from 500ing on a formatting hiccup. `sources` is always rebuilt from the
    # chunks we actually retrieved (ground truth), never the LLM's own field,
    # which it can hallucinate.
    try:
        parsed = RAGResponse(**json.loads(raw))
        parsed.sources = real_sources
        return parsed
    except Exception as e:
        try:
            retry_msg = (
                f"Your previous JSON output had an error: {e}\n"
                f"Original output: {raw}\n"
                f"Return ONLY valid JSON matching the schema."
            )
            retry_result = llm_manager.generate(
                SYSTEM, retry_msg, max_tokens=max_tokens, api_key=api_key
            )
            parsed = RAGResponse(**json.loads(retry_result.text))
            parsed.sources = real_sources
            return parsed
        except Exception:
            log.warning("JSON self-repair failed; returning raw text at low confidence")
            return RAGResponse(
                answer=raw or "Unable to generate answer.",
                sources=real_sources[:3],
                confidence=0.1,
                has_answer=bool(raw),
            )


@traceable(name="ask_stream", run_type="chain")
async def ask_stream(question: str, k: int = 5, owner: str | None = None, api_key: str | None = None):
    """Async streaming RAG query. Yields ("token", text) tuples followed by
    exactly one ("sources", [paths]); SSE framing is the API layer's concern.

    `owner` scopes retrieval to that session's own ingested chunks. `api_key`
    (BYOK) uses that Gemini key instead of the server's own.
    """
    # Retrieval is CPU-bound (embedding + cross-encoder), ~200-400ms. Run it in
    # a worker thread so it doesn't block the event loop — otherwise every
    # other in-flight request (including other streams already sending tokens)
    # stalls for the duration of this one query's retrieval.
    chunks = await run_in_threadpool(lambda: get_retriever().retrieve(question, k=k, owner=owner))

    if not chunks:
        yield "token", _no_documents_message(owner)
        yield "sources", []
        return

    context, sources = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    # Stream for time-to-first-token: an answer that starts rendering at 300ms
    # feels faster than a shorter one that appears all at once.
    t0 = time.time()
    _llm_stats["calls"] += 1
    try:
        async for token in llm_manager.stream(
            STREAM_SYSTEM, user_message, max_tokens=get_settings().llm_max_tokens, api_key=api_key
        ):
            yield "token", token
        _llm_stats["providers_used"][llm_manager.active_provider] = (
            _llm_stats["providers_used"].get(llm_manager.active_provider, 0) + 1
        )
    except Exception:
        _llm_stats["errors"] += 1
        raise
    finally:
        _llm_stats["total_ms"] += (time.time() - t0) * 1000

    # dict.fromkeys dedupes while preserving retrieval order (set() would not).
    yield "sources", list(dict.fromkeys(sources))
