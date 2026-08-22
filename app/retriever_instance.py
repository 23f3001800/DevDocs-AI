"""Shared HybridRetriever singleton — chain.py and main.py reuse one instance
so we don't pay ~400 MB twice (2x CrossEncoder + 2x BM25 index)."""

import logging
import threading
import time

log = logging.getLogger(__name__)

_retriever = None
_lock = threading.Lock()


def get_retriever():
    """Lazy-init the HybridRetriever singleton with double-checked locking
    (hot path stays lock-free; the lock stops two threads double-building it)."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                from app.hybrid_retriever import HybridRetriever

                _retriever = HybridRetriever()
    return _retriever


def warm():
    """Pre-warm the models with a dummy query so the first real query pays no
    cold start (SentenceTransformer load, first embed, first CrossEncoder predict)."""
    t0 = time.time()
    get_retriever().retrieve("warmup query", k=1)
    log.info("Retriever pre-warmed in %.0fms", (time.time() - t0) * 1000)
