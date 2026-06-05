import json, os
from anthropic import Anthropic
from app.hybrid_retriever import HybridRetriever
from app.models import RAGResponse
from dotenv import load_dotenv

load_dotenv()

# WHY module-level singletons? # HybridRetriever loads the CrossEncoder model and BM25 index
# on init — this takes 2-3 seconds. Loading per-request would
# make every query slow. Module-level means one load at startup,
# shared across all requests (safe — both are read-only).
retriever = HybridRetriever()
client = Anthropic()


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


def ask(question: str, k: int = 5) -> RAGResponse:
    # Step 1: Retrieve relevant chunks
    chunks = retriever.retrieve(question, k=k)

    if not chunks:
        return RAGResponse(
            answer="No documents have been ingested yet. Run scripts/ingest.py first.",
            sources=[], confidence=0.0, has_answer=False
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
        context_parts.append(
            f"[Chunk {i+1} | {fp} | relevance: {score:.2f}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    
    # Step 3: Call LLM
    resp = client.messages.create(
        model=os.getenv("MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    raw = resp.content[0].text.strip()

    # Step 4: Parse with retry
    # WHY retry? LLMs occasionally produce malformed JSON.
    # Rather than crashing, we send the error back to Claude
    # and ask it to fix its own output. Works ~99% of the time.
    try:
        return RAGResponse(**json.loads(raw))
    except Exception as e:
        retry = client.messages.create(
            model=os.getenv("MODEL", "claude-sonnet-4-20250514"),
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"Your previous JSON output had an error: {e}\nOriginal output: {raw}\nReturn ONLY valid JSON matching the schema."
            }]
        )
        return RAGResponse(**json.loads(retry.content[0].text.strip()))
    
from anthropic import AsyncAnthropic
from typing import AsyncGenerator

async_client = AsyncAnthropic()

async def ask_async(question: str, k: int = 5) -> AsyncGenerator[RAGResponse, None]:
    chunks = retriever.retrieve(question, k=k)
    if not chunks:
        yield RAGResponse(
            answer="No documents have been ingested yet. Run scripts/ingest.py first.",
            sources=[], confidence=0.0, has_answer=False
        )
        return

    context_parts = []
    for i, chunk in enumerate(chunks):
        fp = chunk["metadata"].get("file_path", "unknown")
        score = chunk["rerank_score"]
        context_parts.append(
            f"[Chunk {i+1} | {fp} | relevance: {score:.2f}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    async for resp in async_client.messages.stream(
        model=os.getenv("MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    ):
        raw = resp.content[0].text.strip()
        try:
            yield RAGResponse(**json.loads(raw))
            return
        except Exception as e:
            retry_resp = await async_client.messages.create(
                model=os.getenv("MODEL", "claude-sonnet-4-20250514"),
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": f"Your previous JSON output had an error: {e}\nOriginal output: {raw}\nReturn ONLY valid JSON matching the schema."
                }]
            )
            retry_raw = retry_resp.content[0].text.strip()
            yield RAGResponse(**json.loads(retry_raw))


async def ask_stream(question: str, k: int = 5) -> AsyncGenerator[str, None]:
    """Async streaming version — yields tokens for FastAPI StreamingResponse."""
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
    async with async_client.messages.stream(
        model=os.getenv("MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1024,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    ) as stream:
        async for text in stream.text_stream:
            yield text

    # Yield sources as a final delimiter after the answer
    # WHY \n\n||SOURCES||? The client splits on this marker
    # to separate the answer text from the sources JSON.
    import json
    yield f"\n\n||SOURCES||{json.dumps(list(set(sources)))}"