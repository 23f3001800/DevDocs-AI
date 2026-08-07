"""
Shared singleton for HybridRetriever.

WHY a separate module? Both chain.py and main.py need the same retriever
instance. If chain.py created one at import time AND main.py created another in
lifespan(), we would waste ~400 MB of RAM (2x CrossEncoder + 2x BM25 index).
"""

import logging
import threading
import time

log = logging.getLogger(__name__)

_retriever = None
_lock = threading.Lock()


def get_retriever():
    """Lazy-init the HybridRetriever singleton (thread-safe).

    Double-checked locking: the outer check keeps the hot path lock-free once
    the instance exists, the inner check (under the lock) stops two racing
    threads from each building a ~400 MB retriever.
    """
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                from app.hybrid_retriever import HybridRetriever

                _retriever = HybridRetriever()
    return _retriever


def warm():
    """Pre-warm all models with a dummy query.

    Forces the SentenceTransformer load, the first embedding computation, and
    the first CrossEncoder predict() — each of which is far slower than every
    subsequent call. After this the first real user query has no cold start.
    """
    t0 = time.time()
    get_retriever().retrieve("warmup query", k=1)
    log.info("Retriever pre-warmed in %.0fms", (time.time() - t0) * 1000)
