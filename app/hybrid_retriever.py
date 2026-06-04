# WHY Reciprocal Rank Fusion (RRF)?
#   - Concat and dedup: doesn't consider rank quality
#   - Average scores: scores are on different scales (cosine vs BM25)
#   - RRF: rank-based, scale-invariant, proven on TREC benchmarks



from app.vectorstore import VectorStore
from app.bm25_retriever import BM25Retriever

class HybridRetriever:
    def __init__(self, k_rrf: int = 60):
        self.vs = VectorStore()
        self.bm25 = BM25Retriever()
        self.k_rrf = k_rrf
        self._build_bm25_from_chroma()

    def _build_bm25_from_chroma(self):
        """Build BM25 index from Chroma's documents."""
        result = self.vs.collection.get(
            include=["documents", "metadatas"]
        )
        if result["documents"]:
            self.bm25.build_index(result["documents"],
                                  result["metadatas"])
            print(f"BM25 index built: {len(result['documents'])} docs")
    
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Hybrid retrieval: dense + BM25 merged with RRF."""
        dense  = self.vs.query(query, k=10)
        sparse = self.bm25.search(query, k=10)

        # Build RRF score map {content: score}
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(dense):
            c = item["content"]
            rrf_scores[c] = rrf_scores.get(c,0) + 1/(self.k_rrf + rank + 1)

        for rank, (doc, meta, _) in enumerate(sparse):
            rrf_scores[doc] = rrf_scores.get(doc,0) + 1/(self.k_rrf + rank + 1)

        # Build metadata map for final results
        meta_map = {item["content"]: item["metadata"] for item in dense}
        for doc, meta, _ in sparse:
            if doc not in meta_map:
                meta_map[doc] = meta

        sorted_docs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [{
            "content": d,
            "metadata": meta_map.get(d, {}),
            "rrf_score": rrf_scores[d]
        } for d in sorted_docs]