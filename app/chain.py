import json
import os
import time

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv

from app.models import RAGResponse
from app.retriever_instance import get_retriever

load_dotenv()

# ── LangSmith tracing (optional) ─────────────────────────────
# WHY conditional import? LangSmith is optional — if the user doesn't
# set LANGCHAIN_API_KEY, everything still works. The @traceable decorator
# is a no-op wrapper when tracing is disabled.
_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
if _tracing_enabled:
    try:
        from langsmith import traceable
    except ImportError:

        def traceable(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator if not args or not callable(args[0]) else args[0]
else:

    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator if not args or not callable(args[0]) else args[0]


# ── Anthropic clients (singletons) ───────────────────────────
client = Anthropic()
async_client = AsyncAnthropic()

# ── Metrics tracking ─────────────────────────────────────────
_llm_stats = {"calls": 0, "errors": 0, "total_ms": 0}


def get_llm_stats() -> dict:
    """Return LLM call stats for /metrics."""
    return {
        "llm_calls": _llm_stats["calls"],
        "llm_errors": _llm_stats["errors"],
        "llm_avg_ms": round(_llm_stats["total_ms"] / _llm_stats["calls"], 1)
        if _llm_stats["calls"] > 0
        else 0,
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


@traceable(name="ask_sync", run_type="chain")
def ask(question: str, k: int = 5) -> RAGResponse:
    """Synchronous RAG query — returns structured RAGResponse."""
    retriever = get_retriever()

    # Step 1: Retrieve relevant chunks
    chunks = retriever.retrieve(question, k=k)

    if not chunks:
        return RAGResponse(
            answer="No documents have been ingested yet. Run scripts/ingest.py first.",
            sources=[],
            confidence=0.0,
            has_answer=False,
        )

    # Step 2: Build context block
    # WHY include file_path and rerank_score in context?
    # Claude uses file_path to populate the sources field accurately.
    # rerank_score helps Claude calibrate confidence — lower scores
    # mean the retrieved chunks are less relevant to this question.
    context_parts = []
    for i, chunk in enumerate(chunks):
        fp = chunk["metadata"].get("file_path", "unknown")
        score = chunk["rerank_score"]
        context_parts.append(f"[Chunk {i+1} | {fp} | relevance: {score:.2f}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Call LLM
    t0 = time.time()
    _llm_stats["calls"] += 1
    try:
        resp = client.messages.create(
            model=os.getenv("MODEL", "claude-sonnet-4-6"),
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
    except Exception:
        _llm_stats["errors"] += 1
        raise
    finally:
        _llm_stats["total_ms"] += (time.time() - t0) * 1000

    raw = resp.content[0].text.strip()

    # Step 4: Parse with retry
    # WHY retry? LLMs occasionally produce malformed JSON.
    # Rather than crashing, we send the error back to Claude
    # and ask it to fix its own output. Works ~99% of the time.
    try:
        return RAGResponse(**json.loads(raw))
    except Exception as e:
        retry = client.messages.create(
            model=os.getenv("MODEL", "claude-sonnet-4-6"),
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": f"Your previous JSON output had an error: {e}\nOriginal output: {raw}\nReturn ONLY valid JSON matching the schema.",
                }
            ],
        )
        return RAGResponse(**json.loads(retry.content[0].text.strip()))


@traceable(name="ask_stream", run_type="chain")
async def ask_stream(question: str, k: int = 5):
    """Async streaming version — yields tokens for FastAPI StreamingResponse."""
    retriever = get_retriever()

    # Retrieval is CPU-bound (embedding + cosine), safe to call sync here
    chunks = retriever.retrieve(question, k=k)

    if not chunks:
        yield "⚠️ No documents ingested yet. Run: python scripts/ingest.py --source <url>"
        return

    # Build context block with source attribution
    parts, sources = [], []
    for chunk in chunks:
        fp = chunk["metadata"].get("file_path", "unknown")
        sources.append(fp)
        parts.append(f"[{fp} | score:{chunk['rerank_score']:.2f}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(parts)

    # WHY stream tokens instead of yielding the full answer?
    # Time-to-first-token (TTFT) is what users perceive as "speed".
    # A 3-second full response feels slow.
    # The same response streamed feels instant because the user
    # sees the first word in ~300ms. TTFT is the key UX metric.
    t0 = time.time()
    _llm_stats["calls"] += 1
    try:
        async with async_client.messages.stream(
            model=os.getenv("MODEL", "claude-sonnet-4-6"),
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except Exception:
        _llm_stats["errors"] += 1
        raise
    finally:
        _llm_stats["total_ms"] += (time.time() - t0) * 1000

    # Yield sources as a final delimiter after the answer
    # WHY \n\n||SOURCES||? The client splits on this marker
    # to separate the answer text from the sources JSON.
    yield f"\n\n||SOURCES||{json.dumps(list(set(sources)))}"
