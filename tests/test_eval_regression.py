from evals.metrics import calculate_mrr, calculate_recall_at_k


def test_retrieval_quality_thresholds():
    # Example test with mock responses to prevent regression in CI
    ground_truth = {"doc_1", "doc_2"}
    retrieved = ["doc_3", "doc_1", "doc_5", "doc_2"]

    recall = calculate_recall_at_k(ground_truth, retrieved, k=3)
    mrr = calculate_mrr(ground_truth, retrieved, k=3)

    # Hard thresholds for retrieval regressions
    assert recall >= 0.5, f"Recall@3 degraded: {recall}"
    assert mrr >= 0.33, f"MRR degraded: {mrr}"
