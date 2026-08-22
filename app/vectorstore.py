import hashlib
import logging
from collections import OrderedDict

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from app.config import get_settings

log = logging.getLogger(__name__)

# ── Embedding Cache ──────────────────────────────────────────
# Embedding is CPU-bound (~50ms); an LRU cache skips recomputing repeat queries.
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
# VectorStore() is built on many hot paths; load the ~90 MB model and the
# Chroma client once per process and share them rather than per instance.
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
    """Chunk count without constructing a VectorStore, so the /health probe
    (every 30s) never triggers the 90 MB embedder load."""
    settings = get_settings()
    collection = _get_client().get_or_create_collection(
        name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )
    return collection.count()


def make_chunk_id(metadata: dict, fallback_index: int = 0) -> str:
    """Deterministic, collision-proof chunk ID.

    Hashes (source, file_path, chunk_index) so URL/PDF sources with no distinct
    file_path can't collapse onto shared "doc_0" ids, and same-named files from
    different repos don't clobber each other. Determinism makes re-ingest
    idempotent. `commit_sha` and `owner` are appended ONLY when present (legacy
    chunks keep their old ids): `commit_sha` makes repo ingests version-tagged
    and non-destructive; `owner` keeps two sessions ingesting the same source
    on independent, per-session-private rows.
    """
    source = str(metadata.get("source", ""))
    file_path = str(metadata.get("file_path", ""))
    chunk_index = metadata.get("chunk_index", fallback_index)
    commit_sha = str(metadata.get("commit_sha", "") or "")
    owner = str(metadata.get("owner", "") or "")
    key = f"{source}\x00{file_path}\x00{chunk_index}"
    if commit_sha:
        key = f"{key}\x00{commit_sha}"
    if owner:
        key = f"{key}\x00{owner}"
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

    def delete_by_source(self, source: str, owner: str | None = None) -> int:
        """Remove chunks that came from `source`. Returns count deleted.

        Needed because upsert never removes anything, so upstream-deleted content
        would stay retrievable; re-ingest is delete-then-insert. `owner` scopes
        the delete to one session's copy so it can't remove another session's
        chunks; `owner=None` (offline scripts) purges every copy.
        """
        where = {"source": source} if owner is None else {"$and": [{"source": source}, {"owner": owner}]}
        existing = self.collection.get(where=where, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
            log.info(
                "Deleted %d chunks for source %s%s",
                len(ids),
                source,
                f" (owner={owner})" if owner else "",
            )
        return len(ids)

    def query(
        self, text: str, k: int = 5, file_type: str | None = None, owner: str | None = None
    ) -> list[dict]:
        """Query by semantic similarity, with optional file_type / owner filters.

        `owner` scopes results to one session's own chunks (legacy chunks with
        no `owner` match nobody, never silently shared); `owner=None` (offline
        scripts) skips the filter and searches every chunk.
        """
        q_embed = [self._cached_encode(text)]
        conditions = []
        if file_type:
            conditions.append({"file_type": file_type})
        if owner:
            conditions.append({"owner": owner})
        where = conditions[0] if len(conditions) == 1 else ({"$and": conditions} if conditions else None)
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
