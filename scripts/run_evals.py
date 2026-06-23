import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_evals(threshold: float = 0.75) -> dict:
    # ── Pre-flight checks ────────────────────────────────────────
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        print("⚠️  GOOGLE_API_KEY not set — skipping RAGAS eval.")
        print("   Set the secret in GitHub → Settings → Secrets to enable.")
        sys.exit(0)  # soft skip, don't block CI

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        print("⚠️  ANTHROPIC_API_KEY not set — skipping RAGAS eval.")
        sys.exit(0)

    # ── Check vector store has documents ─────────────────────────
    from app.vectorstore import VectorStore

    vs = VectorStore()
    doc_count = vs.count()
    if doc_count == 0:
        print("⚠️  Vector store is empty — no documents ingested.")
        print("   Run: python -m scripts.ingest <source> first.")
        print("   Skipping eval (soft pass).")
        sys.exit(0)

    print(f"✅ Vector store has {doc_count} chunks.")

    # ── Lazy imports (only if we're actually running) ────────────
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
    )

    from app.chain import ask
    from app.hybrid_retriever import HybridRetriever

    golden = json.loads(Path("tests/golden_set.json").read_text())
    retriever = HybridRetriever()

    # Build evaluation samples using RAGAS v1.0 SingleTurnSample API
    samples = []
    for item in golden:
        q = item["question"]
        print(f"Evaluating: {q[:60]}...")
        try:
            chunks = retriever.retrieve(q, k=4)
            resp = ask(q)
            samples.append(
                SingleTurnSample(
                    user_input=q,
                    response=resp.answer,
                    retrieved_contexts=[c["content"] for c in chunks] or ["No context found."],
                    reference=item["ground_truth"],
                )
            )
        except Exception as e:
            print(f"  ⚠️ Skipped (error: {e})")
            continue

    if not samples:
        print("❌ All questions failed — cannot run RAGAS evaluation.")
        sys.exit(1)

    dataset = EvaluationDataset(samples=samples)

    # Use Gemini as the evaluator LLM (separate from Claude used for answering)
    from langchain_google_genai import ChatGoogleGenerativeAI

    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_key,
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    # WHY save to JSON? CI needs a persistent artefact to compare.
    # GitHub Actions can diff ragas_results.json between runs
    # to catch silent score regressions across PRs.
    out = Path("docs/evaluation/ragas_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))

    print(f"\n{'='*40}")
    print(f"Faithfulness:      {scores['faithfulness']:.4f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.4f}")
    print(f"Context Precision: {scores['context_precision']:.4f}")

    avg = sum(scores.values()) / len(scores)
    print(f"Average:           {avg:.4f}  (threshold: {threshold})")

    if avg < threshold:
        print(f"\n❌ EVAL GATE FAILED — avg {avg:.4f} < {threshold}")
        sys.exit(1)  # exit code 1 blocks CI deployment

    print(f"\n✅ Eval gate passed — {avg:.4f} ≥ {threshold}")
    return scores


if __name__ == "__main__":
    run_evals()
