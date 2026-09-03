"""Run the golden-set evaluation against the real DevDocs AI pipeline."""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from app.chain import ask
from app.llm_providers import llm_manager
from app.retriever_instance import get_retriever
from evals.metrics import (
    calculate_citation_coverage,
    calculate_hit_rate,
    calculate_keyword_coverage,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

ROOT = Path(__file__).resolve().parents[1]


def run_ragas(records: dict[str, list]) -> dict[str, float]:
    """Run paid RAGAS judge metrics through the application's configured LLM."""
    from datasets import Dataset
    from langchain_core.outputs import Generation, LLMResult
    from ragas import evaluate
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import answer_relevancy, faithfulness

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

    result = evaluate(
        Dataset.from_dict(records),
        metrics=[faithfulness, answer_relevancy],
        llm=PipelineJudge(),
    )
    return {
        "ragas_faithfulness": float(result["faithfulness"]),
        "ragas_answer_relevancy": float(result["answer_relevancy"]),
    }


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Evaluation dataset must contain a non-empty JSON list: {path}")
    required = {"question", "ground_truth_doc_ids", "ground_truth_answer"}
    for index, case in enumerate(cases):
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Case {index} is missing: {', '.join(sorted(missing))}")
    return cases


def run(cases: list[dict], k: int, retrieval_only: bool, use_ragas: bool) -> dict[str, float | int]:
    retriever = get_retriever()
    retrieval = {name: [] for name in ("recall_at_k", "precision_at_k", "mrr", "hit_rate")}
    answer_scores = {name: [] for name in ("keyword_coverage", "citation_coverage")}
    latencies = []
    failures = 0
    ragas_records = {"question": [], "contexts": [], "answer": [], "ground_truth": []}

    for case in cases:
        started = time.perf_counter()
        chunks = retriever.retrieve(case["question"], k=k)
        relevant_ids = set(case["ground_truth_doc_ids"])
        retrieved_ids = [
            next((doc_id for doc_id in relevant_ids if doc_id in chunk["content"]), chunk["id"])
            for chunk in chunks
        ]
        retrieval["recall_at_k"].append(calculate_recall_at_k(relevant_ids, retrieved_ids, k))
        retrieval["precision_at_k"].append(calculate_precision_at_k(relevant_ids, retrieved_ids, k))
        retrieval["mrr"].append(calculate_mrr(relevant_ids, retrieved_ids, k))
        retrieval["hit_rate"].append(calculate_hit_rate(relevant_ids, retrieved_ids, k))

        if not retrieval_only:
            response = ask(case["question"], k=k)
            ragas_records["question"].append(case["question"])
            ragas_records["contexts"].append([chunk["content"] for chunk in chunks])
            ragas_records["answer"].append(response.answer)
            ragas_records["ground_truth"].append(case["ground_truth_answer"])
            answer_scores["keyword_coverage"].append(
                calculate_keyword_coverage(response.answer, case["ground_truth_answer"])
            )
            answer_scores["citation_coverage"].append(
                calculate_citation_coverage(response.answer, response.sources)
            )
            failures += int(not response.has_answer)
        latencies.append((time.perf_counter() - started) * 1000)

    scores = {**retrieval, **answer_scores}
    results = {name: statistics.fmean(values) for name, values in scores.items() if values}
    results["avg_latency_ms"] = statistics.fmean(latencies)
    results["cases"] = len(cases)
    results["answer_failures"] = failures
    if use_ragas:
        results.update(run_ragas(ragas_records))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "eval_dataset.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM answer evaluation")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS LLM-judge metrics")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON output")
    parser.add_argument("--fail-under", type=float, help="Fail if mean quality is lower")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    try:
        if args.ragas and args.retrieval_only:
            parser.error("--ragas requires answer evaluation; remove --retrieval-only")
        results = run(load_cases(args.dataset), args.k, args.retrieval_only, args.ragas)
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
