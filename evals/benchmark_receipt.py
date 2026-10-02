"""Measure actual retrieval on the checked-in fixture; never call a mock LLM."""
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from app.hybrid_retriever import HybridRetriever
from app.vectorstore import VectorStore
from evals.run_evals import (
    _BM25OnlyRetriever,
    _DenseOnlyRetriever,
    _HybridNoRerankRetriever,
    load_cases,
    run,
)

def main():
    output = Path("benchmark-results")
    output.mkdir(exist_ok=True)
    dataset = Path("data/eval_dataset.json")
    corpus = Path("data/devdocs_ragas_eval_test_cases.pdf")
    cases = load_cases(dataset)
    configurations = {
        "Dense": _DenseOnlyRetriever(),
        "BM25": _BM25OnlyRetriever(),
        "Hybrid": _HybridNoRerankRetriever(),
        "Hybrid + Reranking": HybridRetriever(),
    }
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "cases": len(cases),
        "chunks": VectorStore().count(),
        "k": 5,
        "generation_evaluated": False,
        "scope": "Existing labelled PDF fixture; not held-out cross-repository evaluation",
        "metric_note": "Document labels are matched inside fixture chunks using the existing evaluator. Inspect per-case contexts for label-matching limitations.",
        "configurations": {},
    }
    for name, retriever in configurations.items():
        runs = []
        for iteration in range(3):
            details = []
            class Recorder:
                def retrieve(self, question, k):
                    start = time.perf_counter()
                    chunks = retriever.retrieve(question, k=k)
                    details.append({
                        "question": question,
                        "retrieval_ms": (time.perf_counter() - start) * 1000,
                        "chunks": chunks,
                    })
                    return chunks
            scores = run(cases, 5, True, False, retriever=Recorder())
            scores.pop("answer_failures", None)
            runs.append({"iteration": iteration + 1, "metrics": scores, "cases": details})
        latencies = sorted(case["retrieval_ms"] for item in runs[1:] for case in item["cases"])
        receipt["configurations"][name] = {
            "runs": runs,
            "warm_p50_ms": statistics.median(latencies),
            "warm_p95_ms": latencies[max(0, int(.95 * len(latencies)) - 1)],
        }
        (output / "retrieval.json").write_text(json.dumps(receipt, indent=2, default=str))
        print(json.dumps({"configuration": name, "runs": [item["metrics"] for item in runs]}), flush=True)
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output / "retrieval.json").write_text(json.dumps(receipt, indent=2, default=str))
    print("BENCHMARK_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
