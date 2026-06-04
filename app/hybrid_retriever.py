# WHY Reciprocal Rank Fusion (RRF)?
#   - Concat and dedup: doesn't consider rank quality
#   - Average scores: scores are on different scales (cosine vs BM25)
#   - RRF: rank-based, scale-invariant, proven on TREC benchmarks

from sentence_transformers import CrossEncoder
import time

from app.vectorstore import VectorStore
from app.bm25_retriever import BM25Retriever

class HybridRetriever:
    def __init__(self, k_rrf: int = 60):
        self.vs = VectorStore()
        self.bm25 = BM25Retriever()
        self.k_rrf = k_rrf
        self._build_bm25_from_chroma()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _build_bm25_from_chroma(self):
        """Build BM25 index from Chroma's documents."""
        result = self.vs.collection.get(
            include=["documents", "metadatas"]
        )
        if result["documents"]:
            self.bm25.build_index(result["documents"],
                                  result["metadatas"])
            print(f"BM25 index built: {len(result['documents'])} docs")
    
    def retrieve(self, query: str, k: int = 5, initial_k: int = 20) -> list[dict]:
        """ 3-stage pipeline:
                1. Dense + BM25 → top-20 candidates (fast, coarse)
                2. RRF merge → unified ranking
                3. CrossEncoder → top-5 final results (slow, precise)
        """
        t0 = time.time()
        dense  = self.vs.query(query, k=initial_k)
        sparse = self.bm25.search(query, k=initial_k)

        # Stage 1: RRF merge (same as before)
        rrf_scores: dict[str, float] = {}
        meta_map: dict[str, dict] = {}

        for rank, item in enumerate(dense):
            c = item["content"]
            rrf_scores[c] = rrf_scores.get(c,0) + 1/(60+rank+1)
            meta_map[c] = item["metadata"]
        for rank, (doc, meta, _) in enumerate(sparse):
            rrf_scores[doc] = rrf_scores.get(doc,0) + 1/(60+rank+1)
            if doc not in meta_map: meta_map[doc] = meta

        candidates = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:initial_k]
        print(f"RRF merged candidates: {len(candidates)} unique docs from dense+BM25")

        # Stage 2: CrossEncoder reranking
        # WHY pairs = [[query, doc], [query, doc], ...]?
        # CrossEncoder expects query-document pairs because it processes
        # them together with full attention.
        # predict() returns a relevance score for each pair.
        # Higher score = more relevant to this specific query.
        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        t1 = time.time()
        print(f"Hybrid+rerank: {len(candidates)} candidates → {k} results in {(t1-t0)*1000:.0f}ms")
        print("Top candidates after RRF (before reranking):")
        for doc, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  RRF score={score:.4f} doc='{doc[:50]}...'")

        return [{
            "content": doc,
            "metadata": meta_map.get(doc, {}),
            "rerank_score": float(score)
        } for doc, score in reranked]