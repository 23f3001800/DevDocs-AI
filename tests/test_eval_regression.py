from pathlib import Path

from evals.metrics import calculate_mrr, calculate_precision_at_k, calculate_recall_at_k
from evals.run_evals import _retrieved_doc_id, load_cases


def test_retrieval_quality_thresholds():
    # Example test with mock responses to prevent regression in CI
    ground_truth = {"doc_1", "doc_2"}
    retrieved = ["doc_3", "doc_1", "doc_5", "doc_2"]

    recall = calculate_recall_at_k(ground_truth, retrieved, k=3)
    mrr = calculate_mrr(ground_truth, retrieved, k=3)

    # Hard thresholds for retrieval regressions
    assert recall >= 0.5, f"Recall@3 degraded: {recall}"
    assert mrr >= 0.33, f"MRR degraded: {mrr}"


def test_precision_at_k_keeps_requested_denominator():
    assert calculate_precision_at_k({"doc_a"}, ["doc_a", "doc_a", "doc_b"], k=3) == 1 / 3


def test_eval_identity_comes_from_metadata_not_expected_answer():
    chunk = {
        "id": "chunk-1",
        "content": "This text mentions doc_wrong_99.",
        "metadata": {"doc_id": "doc_right_01"},
    }
    assert _retrieved_doc_id(chunk) == "doc_right_01"


def test_extended_eval_dataset_contains_25_unique_cases():
    cases = load_cases(Path(__file__).parents[1] / "data" / "eval_dataset_25.json")
    assert len(cases) == 25
    assert len({case["case_id"] for case in cases}) == 25
