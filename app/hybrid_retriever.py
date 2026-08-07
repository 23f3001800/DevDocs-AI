"""Three-stage retrieval funnel: recall → fusion → precision.

    ~N chunks
        │
   ┌────┴────┐        Stage 1 RECALL    (cheap, wide)  → top-`initial_k` each
   ▼         ▼
 Dense     BM25
   └────┬────┘        Stage 2 FUSION    (free)         → RRF merge
        ▼
  CrossEncoder        Stage 3 PRECISION (expensive)    → top-k
        ▼
      top-k → LLM

The principle: cheap-and-wide first, expensive-and-narrow last. Running the
cross-encoder over the whole corpus would take minutes; over ~20 candidates it
takes ~200ms.
"""

import logging
import threading
import time

from sentence_transformers import CrossEncoder

from app.bm25_retriever import BM25Retriever
from app.config import get_settings
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, k_rrf: int | None = None):
        settings = get_settings()
        # WHY Reciprocal Rank Fusion? Dense returns cosine scores in [0,1];
        # BM25 returns unbounded scores. You cannot average or add them — the
        # scales differ and shift per query. RRF discards scores and uses only
        # rank, making it scale-invariant and tuning-free.
        self.k_rrf = settings.rrf_k if k_rrf is None else k_rrf
        self.default_initial_k = settings.retrieval_initial_k
        self.vs = VectorStore()
        self.bm25 = BM25Retriever()
        # Guards rebuilds triggered by /ingest while queries are in flight.
        self._bm25_lock = threading.Lock()
        self.rebuild_bm25()
        self.reranker = CrossEncoder(settings.reranker_model)

    def rebuild_bm25(self) -> int:
        """(Re)build the BM25 index from Chroma. Returns the document count.

        WHY this is public and called after ingest: the index is in-memory and
        was previously built exactly once, in __init__. Because the retriever is
        a process-wide singleton, every chunk added by /ingest afterwards was
        visible to dense search (which queries Chroma live) but invisible to
        sparse search until the process restarted — so newly ingested exact
        identifiers, the very thing BM25 exists for, could not be found.
        """
        result = self.vs.all_documents()
        documents = result.get("documents") or []
        with self._bm25_lock:
            self.bm25.build_index(
                documents,
                result.get("metadatas") or [],
                result.get("ids") or None,
            )
        log.info("BM25 index built: %d docs", len(documents))
        return len(documents)

    def retrieve(self, query: str, k: int = 5, initial_k: int | None = None) -> list[dict]:
        """Dense + BM25 → RRF merge → cross-encoder rerank → top-k."""
        t0 = time.time()
        initial_k = initial_k or self.default_initial_k

        dense = self.vs.query(query, k=initial_k)
        with self._bm25_lock:
            sparse = self.bm25.search(query, k=initial_k)

        # ── Stage 2: RRF merge ───────────────────────────────
        # Keyed on chunk ID, not document text: hashing full documents is
        # wasteful, and two chunks with identical content (shared boilerplate,
        # duplicated licence headers) would collapse into one entry carrying
        # whichever metadata arrived first — so a citation could name the wrong
        # file.
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for hits in (dense, sparse):
            for rank, item in enumerate(hits):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
                by_id.setdefault(doc_id, item)

        if not rrf_scores:
            return []

        candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:initial_k]

        # ── Stage 3: cross-encoder rerank ────────────────────
        # A bi-encoder must compress a document into a vector without knowing
        # the question. A cross-encoder sees [query, doc] together with full
        # cross-attention, so it is markedly more accurate — and cannot be
        # precomputed, which is why it only runs on the shortlist.
        pairs = [[query, by_id[doc_id]["content"]] for doc_id in candidate_ids]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:k]

        log.info(
            "Hybrid retrieve: dense=%d sparse=%d → %d candidates → %d results in %.0fms",
            len(dense),
            len(sparse),
            len(candidate_ids),
            len(reranked),
            (time.time() - t0) * 1000,
        )
        # Document *content* is deliberately not logged: ingested material would
        # otherwise flow into the log aggregator on every single query.
        log.debug(
            "Top RRF candidates: %s",
            [(doc_id, round(rrf_scores[doc_id], 4)) for doc_id in candidate_ids[:5]],
        )

        return [
            {
                "id": doc_id,
                "content": by_id[doc_id]["content"],
                "metadata": by_id[doc_id].get("metadata", {}),
                "rerank_score": float(score),
            }
            for doc_id, score in reranked
        ]
