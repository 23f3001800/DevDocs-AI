"""
Shared singleton for HybridRetriever.

WHY a separate module? Both chain.py and main.py need the same retriever
instance. If chain.py creates one at import time AND main.py creates another
in lifespan(), we waste ~400MB RAM (2x CrossEncoder + 2x BM25 index).
This module provides one instance, initialized once, used everywhere.
"""
import threading
import time

_retriever = None
_lock = threading.Lock()


def get_retriever():
    """Lazy-init the HybridRetriever singleton (thread-safe)."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:  # double-check locking
                from app.hybrid_retriever import HybridRetriever
                _retriever = HybridRetriever()
    return _retriever


def warm():
    """Pre-warm all models by running a dummy query.

    This forces:
    1. SentenceTransformer model load (if not already loaded)
    2. First embedding computation (triggers ONNX/torch warmup)
    3. CrossEncoder warmup (first predict() is always slow)

    After this, the first real user query has zero cold-start latency.
    """
    t0 = time.time()
    retriever = get_retriever()
    # Run a dummy query to warm the embedding model + CrossEncoder
    _ = retriever.retrieve("warmup query", k=1)
    ms = (time.time() - t0) * 1000
    print(f"[startup] Retriever pre-warmed in {ms:.0f}ms")
