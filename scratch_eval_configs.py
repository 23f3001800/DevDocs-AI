import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
from app import retriever_instance
from app.vectorstore import VectorStore
from app.bm25_retriever import BM25Retriever
from app.hybrid_retriever import HybridRetriever
from evals.run_evals import run, load_cases

os.system("python scripts/ingest.py --source data/devdocs_ragas_eval_test_cases.pdf")
cases = load_cases(Path("data/eval_dataset.json"))

class DenseWrapper:
    def retrieve(self, query, k=5, owner=None):
        chunks = VectorStore().query(query, k=k, owner=owner)
        for c in chunks: c["rerank_score"] = 1.0
        return chunks

class BM25Wrapper:
    def __init__(self):
        self.bm25 = BM25Retriever()
        vs = VectorStore()
        result = vs.all_documents()
        documents = result.get("documents") or []
        self.bm25.build_index(
            documents,
            result.get("metadatas") or [],
            result.get("ids") or None,
        )
    def retrieve(self, query, k=5, owner=None):
        chunks = self.bm25.search(query, k=k, owner=owner)
        for c in chunks: c["rerank_score"] = 1.0
        return chunks

class HybridNoRerankWrapper(HybridRetriever):
    def retrieve(self, query, k=5, initial_k=None, owner=None):
        initial_k = initial_k or self.default_initial_k
        dense = self.vs.query(query, k=initial_k, owner=owner)
        with self._bm25_lock:
            sparse = self.bm25.search(query, k=initial_k, owner=owner)
        rrf_scores = {}
        by_id = {}
        for hits in (dense, sparse):
            for rank, item in enumerate(hits):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
                by_id.setdefault(doc_id, item)
        candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [
            {
                "id": doc_id,
                "content": by_id[doc_id]["content"],
                "metadata": by_id[doc_id].get("metadata", {}),
                "rerank_score": rrf_scores[doc_id],
            }
            for doc_id in candidate_ids
        ]

configs = {
    "Dense": DenseWrapper(),
    "BM25": BM25Wrapper(),
    "Hybrid": HybridNoRerankWrapper(),
    "Hybrid + reranking": HybridRetriever(),
}

results = {}
for name, retriever in configs.items():
    print(f"Running {name}...")
    retriever_instance._retriever = retriever
    res = run(cases, k=5, retrieval_only=False, use_ragas=False)
    results[name] = res

print("\n--- RESULTS ---\n")
print("Configuration | Recall | Hit Rate | MRR | Keyword Coverage | Citation Coverage | Latency (ms)")
for name, res in results.items():
    print(f"{name} | {res['recall_at_k']:.3f} | {res['hit_rate']:.3f} | {res['mrr']:.3f} | {res['keyword_coverage']:.3f} | {res['citation_coverage']:.3f} | {res['avg_latency_ms']:.0f}")
