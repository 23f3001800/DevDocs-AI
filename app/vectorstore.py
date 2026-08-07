import hashlib
import logging
from collections import OrderedDict

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from app.config import get_settings

log = logging.getLogger(__name__)

# ── Embedding Cache ──────────────────────────────────────────
# WHY cache embeddings? The same query often appears multiple times
# (retries, similar questions, eval runs). Embedding is CPU-bound
# (~50ms per query). An LRU cache avoids redundant computation.
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
        _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=get_settings().chroma_path,
            # Off by default: it phones home on every client/collection/query
            # event, and the bundled posthog client errors noisily on each one,
            # burying real ERROR lines in the startup log.
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
    return _client


def collection_count() -> int:
    """Chunk count without constructing a VectorStore.

    WHY separate? VectorStore.__init__ touches the embedder, and /health is
    probed every 30s by the Docker HEALTHCHECK. A liveness probe must not be
    able to trigger a 90 MB model load.
    """
    settings = get_settings()
    collection = _get_client().get_or_create_collection(
        name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )
    return collection.count()


def make_chunk_id(metadata: dict, fallback_index: int = 0) -> str:
    """Deterministic, collision-proof ID for a chunk.

    WHY hash (source, file_path, chunk_index) instead of the old
    f"{file_path}_{chunk_index}"? Because `file_path` used to be absent for URL
    and PDF sources, so every one of them produced the IDs "doc_0", "doc_1", …
    — ingesting a second URL silently *overwrote* the first one's chunks.
    Including `source` also keeps two repos that share a file name (README.md)
    from clobbering each other.

    Determinism is what makes re-ingesting the same content idempotent rather
    than duplicative.
    """
    source = str(metadata.get("source", ""))
    file_path = str(metadata.get("file_path", ""))
    chunk_index = metadata.get("chunk_index", fallback_index)
    key = f"{source}\x00{file_path}\x00{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class VectorStore:
    def __init__(self, collection_name: str | None = None):
        settings = get_settings()
        self.embedder = _get_embedder()
        self.client = _get_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def _cached_encode(self, text: str) -> list[float]:
        """Encode a single text with an LRU cache."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in _embed_cache:
            _cache_stats["hits"] += 1
            _embed_cache.move_to_end(key)  # mark as recently used
            return _embed_cache[key]

        _cache_stats["misses"] += 1
        embedding = self.embedder.encode([text]).tolist()[0]
        cache_max = get_settings().embed_cache_max
        if cache_max > 0:
            _embed_cache[key] = embedding
            while len(_embed_cache) > cache_max:
                _embed_cache.popitem(last=False)  # evict least-recently-used
        return embedding

    def upsert(self, docs: list[Document]) -> int:
        """Embed and store documents. Returns count added."""
        if not docs:
            return 0
        texts = [d.page_content for d in docs]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, batch_size=64).tolist()
        ids = [make_chunk_id(d.metadata, i) for i, d in enumerate(docs)]

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

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk that came from `source`. Returns count deleted.

        WHY this exists: upsert alone never removes anything, so a file deleted
        upstream (or a chunk index that shrank when a file got shorter) would
        stay retrievable forever and the assistant would answer from code that
        no longer exists. Re-ingest is delete-then-insert for this reason.
        """
        existing = self.collection.get(where={"source": source}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
            log.info("Deleted %d chunks for source %s", len(ids), source)
        return len(ids)

    def query(self, text: str, k: int = 5, file_type: str | None = None) -> list[dict]:
        """Query by semantic similarity, with an optional file_type filter."""
        q_embed = [self._cached_encode(text)]
        where = {"file_type": file_type} if file_type else None
        results = self.collection.query(
            query_embeddings=q_embed,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return [
            {
                # Chroma always returns ids; downstream fusion keys on them
                # rather than on the document text.
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]

    def all_documents(self) -> dict:
        """Fetch every chunk (id, text, metadata) — used to build the BM25 index."""
        return self.collection.get(include=["documents", "metadatas"])

    def count(self) -> int:
        return self.collection.count()
