"""Run the golden-set evaluation against the real DevDocs AI pipeline."""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

from evals.metrics import (
    calculate_citation_coverage,
    calculate_hit_rate,
    calculate_keyword_coverage,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_ttft,
)

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = re.compile(r"\bdoc_[a-z]+_\d{2}\b")


# ── RAGAS LLM-judge integration ─────────────────────────────────────────────

def run_ragas(records: dict[str, list]) -> dict[str, float]:
    """Run paid RAGAS judge metrics through the application's configured LLM."""
    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from langchain_core.outputs import Generation, LLMResult
    from ragas import evaluate
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import answer_relevancy, faithfulness

    from app.llm_providers import llm_manager
    from app.vectorstore import VectorStore

    class PipelineJudge(BaseRagasLLM):
        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            text = llm_manager.generate(
                "You are a strict RAG evaluation judge. Return only the format requested.",
                prompt.to_string(),
                max_tokens=2048,
            ).text
            return LLMResult(generations=[[Generation(text=text)] for _ in range(n)])

        async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return await asyncio.to_thread(
                self.generate_text, prompt, n, temperature, stop, callbacks
            )

        def is_finished(self, response):
            return True

    class PipelineEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return VectorStore()._embedder.encode(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return VectorStore()._embedder.encode([text])[0].tolist()

    result = evaluate(
        Dataset.from_dict(records),
        metrics=[faithfulness, answer_relevancy],
        llm=PipelineJudge(),
        embeddings=PipelineEmbeddings(),
    )
    return {
        "ragas_faithfulness": float(result["faithfulness"]),
        "ragas_answer_relevancy": float(result["answer_relevancy"]),
    }


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if isinstance(cases, dict) and "extends" in cases:
        base_path = (path.parent / cases["extends"]).resolve()
        extension = cases.get("cases", [])
        if base_path == path.resolve():
            raise ValueError(f"Evaluation dataset cannot extend itself: {path}")
        cases = [*load_cases(base_path), *extension]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Evaluation dataset must contain a non-empty JSON list: {path}")
    required = {"question", "ground_truth_doc_ids", "ground_truth_answer"}
    seen_case_ids = set()
    for index, case in enumerate(cases):
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {index} is missing: {', '.join(sorted(missing))}")
        case_id = case.get("case_id")
        if not case_id or case_id in seen_case_ids:
            raise ValueError(f"Case {index} has a missing or duplicate case_id: {case_id!r}")
        seen_case_ids.add(case_id)
    return cases


def _retrieved_doc_id(chunk: dict) -> str:
    """Resolve corpus identity independently from the expected answer.

    Clean in-memory fixtures carry ``metadata.doc_id``.  The regex fallback is
    retained for older persisted fixtures, but searches all corpus IDs rather
    than only the expected IDs.  That prevents an irrelevant result from being
    mislabeled merely because the evaluator already knows the answer key.
    """
    metadata = chunk.get("metadata") or {}
    if metadata.get("doc_id"):
        return str(metadata["doc_id"])
    match = DOC_ID.search(chunk.get("content", ""))
    return match.group(0) if match else str(chunk.get("id", "unknown"))


# ── Core evaluation runner ───────────────────────────────────────────────────

def run(
    cases: list[dict],
    k: int,
    retrieval_only: bool,
    use_ragas: bool,
    retriever=None,
    include_case_results: bool = False,
) -> dict[str, float | int | list[dict]]:
    if retriever is None:
        from app.retriever_instance import get_retriever

        retriever = get_retriever()
    retrieval = {name: [] for name in ("recall_at_k", "precision_at_k", "mrr", "hit_rate")}
    answer_scores = {name: [] for name in ("keyword_coverage", "citation_coverage")}
    latencies = []
    failures = 0
    hallucinations = 0
    answerability = []
    retrieval_cases = 0
    ragas_records = {"question": [], "contexts": [], "answer": [], "ground_truth": []}
    case_results = []

    for case in cases:
        started = time.perf_counter()
        chunks = retriever.retrieve(case["question"], k=k)
        relevant_ids = set(case["ground_truth_doc_ids"])
        retrieved_ids = [_retrieved_doc_id(chunk) for chunk in chunks]
        # An intentionally unanswerable case has no relevant IDs. Mixing its
        # forced zero into positive-retrieval metrics makes Recall/Precision/
        # MRR look worse while failing to measure the behavior we actually care
        # about: abstention. Score it separately under answerability instead.
        if relevant_ids:
            retrieval_cases += 1
            retrieval["recall_at_k"].append(calculate_recall_at_k(relevant_ids, retrieved_ids, k))
            retrieval["precision_at_k"].append(
                calculate_precision_at_k(relevant_ids, retrieved_ids, k)
            )
            retrieval["mrr"].append(calculate_mrr(relevant_ids, retrieved_ids, k))
            retrieval["hit_rate"].append(calculate_hit_rate(relevant_ids, retrieved_ids, k))
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "relevant_ids": sorted(relevant_ids),
                    "retrieved_ids": retrieved_ids[:k],
                    "recall_at_k": retrieval["recall_at_k"][-1],
                    "precision_at_k": retrieval["precision_at_k"][-1],
                    "mrr": retrieval["mrr"][-1],
                }
            )

        if not retrieval_only:
            from app.chain import ask

            # Generate from the same retrieval strategy being evaluated. The
            # previous comparison path always re-retrieved with the global
            # Hybrid+Reranking singleton, so its four generation rows did not
            # actually compare four configurations.
            response = ask(
                case["question"],
                k=k,
                retriever=retriever,
                retrieved_chunks=chunks,
            )
            ragas_records["question"].append(case["question"])
            ragas_records["contexts"].append([chunk["content"] for chunk in chunks])
            ragas_records["answer"].append(response.answer)
            ragas_records["ground_truth"].append(case["ground_truth_answer"])
            answer_scores["keyword_coverage"].append(
                calculate_keyword_coverage(response.answer, case["ground_truth_answer"])
            )
            answer_scores["citation_coverage"].append(
                calculate_citation_coverage(
                    response.answer,
                    response.sources,
                    has_answer=response.has_answer,
                )
            )
            expected_answer = bool(relevant_ids)
            answerability.append(float(response.has_answer == expected_answer))
            failures += int(expected_answer and not response.has_answer)
            hallucinations += int(not expected_answer and response.has_answer)
        latencies.append((time.perf_counter() - started) * 1000)

    scores = {**retrieval, **answer_scores}
    results = {name: statistics.fmean(values) for name, values in scores.items() if values}
    results["avg_latency_ms"] = statistics.fmean(latencies)
    results["cases"] = len(cases)
    results["retrieval_cases"] = retrieval_cases
    results["unanswerable_cases"] = len(cases) - retrieval_cases
    results["answer_failures"] = failures
    results["unsupported_answers"] = hallucinations
    if answerability:
        results["answerability_accuracy"] = statistics.fmean(answerability)
    if use_ragas:
        results.update(run_ragas(ragas_records))
    if include_case_results:
        results["case_results"] = case_results
    return results


# ── Multi-configuration comparison ──────────────────────────────────────────
# Wrappers isolate a single retrieval strategy so we can benchmark each one
# independently against the same golden set.


class _DenseOnlyRetriever:
    """Use only dense (vector) search — no BM25, no reranking."""

    def retrieve(self, query, k=5, owner=None):
        from app.vectorstore import VectorStore

        chunks = VectorStore().query(query, k=k, owner=owner)
        for c in chunks:
            c["rerank_score"] = 1.0
        return chunks


class _BM25OnlyRetriever:
    """Use only BM25 sparse search — no dense, no reranking."""

    def __init__(self):
        from app.bm25_retriever import BM25Retriever
        from app.vectorstore import VectorStore

        self.bm25 = BM25Retriever()
        vs = VectorStore()
        result = vs.all_documents()
        self.bm25.build_index(
            result.get("documents") or [],
            result.get("metadatas") or [],
            result.get("ids") or None,
        )

    def retrieve(self, query, k=5, owner=None):
        chunks = self.bm25.search(query, k=k, owner=owner)
        for c in chunks:
            c["rerank_score"] = 1.0
        return chunks


class _HybridNoRerankRetriever:
    """Dense + BM25 fused via RRF, but skip the cross-encoder reranking step."""

    def __init__(self):
        from app.hybrid_retriever import HybridRetriever

        self._hr = HybridRetriever()

    def retrieve(self, query, k=5, owner=None):
        initial_k = self._hr.default_initial_k
        dense = self._hr.vs.query(query, k=initial_k, owner=owner)
        with self._hr._bm25_lock:
            sparse = self._hr.bm25.search(query, k=initial_k, owner=owner)
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for hits in (dense, sparse):
            for rank, item in enumerate(hits):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._hr.k_rrf + rank + 1)
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


def run_compare(cases: list[dict], k: int, retrieval_only: bool, use_ragas: bool) -> dict[str, dict]:
    """Run evaluation across all four retrieval configurations and return a
    ``{config_name: results_dict}`` mapping."""
    from app.hybrid_retriever import HybridRetriever

    configs = {
        "Dense": _DenseOnlyRetriever(),
        "BM25": _BM25OnlyRetriever(),
        "Hybrid": _HybridNoRerankRetriever(),
        "Hybrid + Reranking": HybridRetriever(),
    }
    all_results = {}
    for name, retriever in configs.items():
        print(f"Running {name}...")
        all_results[name] = run(
            cases, k=k, retrieval_only=retrieval_only,
            use_ragas=use_ragas, retriever=retriever,
        )
    return all_results


# ── TTFT measurement ────────────────────────────────────────────────────────

def run_ttft(question: str, k: int = 5, runs: int = 1) -> dict[str, float]:
    """Measure streaming time-to-first-token over *runs* iterations."""
    from app.chain import ask_stream

    ttfts = []
    for _ in range(runs):
        ttfts.append(asyncio.run(calculate_ttft(ask_stream, question, k=k)))
    return {
        "ttft_avg_ms": statistics.fmean(ttfts),
        "ttft_min_ms": min(ttfts),
        "ttft_max_ms": max(ttfts),
        "ttft_runs": runs,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "eval_dataset.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM answer evaluation")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS LLM-judge metrics")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")
    parser.add_argument("--fail-under", type=float, help="Fail if mean quality is lower")
    parser.add_argument(
        "--compare", action="store_true",
        help="Run all four retrieval configs (Dense, BM25, Hybrid, Hybrid+Rerank) side-by-side",
    )
    parser.add_argument(
        "--ttft", action="store_true",
        help="Measure streaming time-to-first-token",
    )
    parser.add_argument("--ttft-runs", type=int, default=3, help="Number of TTFT iterations (default: 3)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    try:
        if args.ragas and args.retrieval_only:
            parser.error("--ragas requires answer evaluation; remove --retrieval-only")

        cases = load_cases(args.dataset)

        # ── Compare mode: run all four configs side-by-side ──────────────
        if args.compare:
            all_results = run_compare(cases, args.k, args.retrieval_only, args.ragas)
            if args.as_json:
                print(json.dumps(all_results, indent=2, sort_keys=True))
            else:
                header = "Configuration | Recall@K | Hit Rate | MRR | Keyword Cov. | Citation Cov. | Latency (ms)"
                print(f"\n{'=' * len(header)}")
                print(header)
                print(f"{'-' * len(header)}")
                for name, res in all_results.items():
                    print(
                        f"{name:25s} | {res.get('recall_at_k', 0):.3f}    | {res.get('hit_rate', 0):.3f}    "
                        f"| {res.get('mrr', 0):.3f} | {res.get('keyword_coverage', 0):.3f}        "
                        f"| {res.get('citation_coverage', 0):.3f}         | {res.get('avg_latency_ms', 0):.0f}"
                    )
                print(f"{'=' * len(header)}\n")
            return 0

        # ── Standard single-config eval ──────────────────────────────────
        results = run(cases, args.k, args.retrieval_only, args.ragas)

        if args.ttft:
            sample_question = cases[0]["question"] if cases else "What is FastAPI?"
            ttft_results = run_ttft(sample_question, k=args.k, runs=args.ttft_runs)
            results.update(ttft_results)

    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print("Evaluation summary")
        for name, value in results.items():
            print(f"{name:22} {value:.3f}" if isinstance(value, float) else f"{name:22} {value}")

    if args.fail_under is not None:
        quality_names = {"recall_at_k", "mrr", "keyword_coverage", "citation_coverage"}
        quality = statistics.fmean(
            value for name, value in results.items() if name in quality_names
        )
        if quality < args.fail_under:
            print(f"Quality gate failed: {quality:.3f} < {args.fail_under:.3f}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
