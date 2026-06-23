import json, os, sys
from pathlib import Path
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

from app.chain import ask
from app.hybrid_retriever import HybridRetriever
from dotenv import load_dotenv
load_dotenv()


def run_evals(threshold: float = 0.75) -> dict:
    golden = json.loads(Path("tests/golden_set.json").read_text())
    retriever = HybridRetriever()

    # Build evaluation samples using RAGAS v1.0 SingleTurnSample API
    samples = []
    for item in golden:
        q = item["question"]
        print(f"Evaluating: {q[:60]}...")
        chunks = retriever.retrieve(q, k=4)
        resp   = ask(q)

        samples.append(SingleTurnSample(
            user_input=q,
            response=resp.answer,
            retrieved_contexts=[c["content"] for c in chunks],
            reference=item["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=samples)

    # Use Gemini as the evaluator LLM (separate from Claude used for answering)
    from langchain_google_genai import ChatGoogleGenerativeAI
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
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
        "faithfulness":      round(float(result["faithfulness"]),      4),
        "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
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
