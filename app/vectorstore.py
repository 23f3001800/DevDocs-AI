import hashlib
import os
from collections import OrderedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Embedding Cache ──────────────────────────────────────────
# WHY cache embeddings? The same query often appears multiple times
# (retries, similar questions, eval runs). Embedding is CPU-bound
# (~50ms per query). An LRU cache avoids redundant computation.
_CACHE_MAX = 512
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_cache_stats = {"hits": 0, "misses": 0}


def get_cache_stats() -> dict:
    """Return embedding cache hit/miss stats for /metrics."""
    total = _cache_stats["hits"] + _cache_stats["misses"]
    return {
        "embedding_cache_hits": _cache_stats["hits"],
        "embedding_cache_misses": _cache_stats["misses"],
        "embedding_cache_hit_rate": round(_cache_stats["hits"] / total, 4) if total > 0 else 0.0,
        "embedding_cache_size": len(_embed_cache),
    }


# ── Shared, lazily-loaded singletons ─────────────────────────
# WHY? VectorStore() is constructed on every /health probe, every /ingest,
# and inside HybridRetriever. Building a SentenceTransformer each time
# re-initialises a ~90 MB torch model — a big, needless cost on a hot path
# (the Docker HEALTHCHECK alone hits it every 30s). Load the model and the
# Chroma client once per process and share them across all instances.
_embedder: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    return _embedder


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
    return _client


class VectorStore:
    def __init__(self, collection_name: str = "devdocs"):
        self.embedder = _get_embedder()
        self.client = _get_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _cached_encode(self, text: str) -> list[float]:
        """Encode a single text with LRU cache."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in _embed_cache:
            _cache_stats["hits"] += 1
            _embed_cache.move_to_end(key)  # mark as recently used
            return _embed_cache[key]

        _cache_stats["misses"] += 1
        embedding = self.embedder.encode([text]).tolist()[0]
        _embed_cache[key] = embedding
        if len(_embed_cache) > _CACHE_MAX:
            _embed_cache.popitem(last=False)  # evict oldest
        return embedding

    def upsert(self, docs: list[Document]) -> int:
        """Embed and store documents. Returns count added."""
        texts = [d.page_content for d in docs]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=64).tolist()
        ids = [
            f"{d.metadata.get('file_path','doc')}_{d.metadata.get('chunk_index',i)}"
            for i, d in enumerate(docs)
        ]
        # Sanitise metadata — ChromaDB only accepts str/int/float/bool
        metadatas = []
        for d in docs:
            clean = {
                k: (str(v) if not isinstance(v, str | int | float | bool) else v)
                for k, v in d.metadata.items()
            }
            metadatas.append(clean)

        self.collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        return len(docs)

    def query(self, text: str, k: int = 5, file_type: str = None) -> list[dict]:
        """Query by semantic similarity, optional file_type filter."""
        q_embed = [self._cached_encode(text)]
        where = {"file_type": file_type} if file_type else None
        results = self.collection.query(
            query_embeddings=q_embed,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]

    def count(self) -> int:
        return self.collection.count()
