"""Run with python -m unittest discover -s tests_offline -v; no model downloads."""
import unittest
from types import SimpleNamespace

from evals.metrics import calculate_source_membership
from evals.run_evals import quality_gate, run


class EvaluationTests(unittest.TestCase):
    def test_answer_uses_the_measured_context_once(self):
        chunks = [{"id": "doc", "content": "alpha", "metadata": {"file_path": "a.md"}}]
        calls = []
        class Retriever:
            def retrieve(self, question, k):
                calls.append(question)
                return chunks
        def answer(question, received):
            self.assertIs(received, chunks)
            return SimpleNamespace(answer="alpha", sources=["a.md"], has_answer=True)
        result = run([{"question": "q", "ground_truth_doc_ids": ["doc"], "ground_truth_answer": "alpha"}], 1, False, False, Retriever(), answer)
        self.assertEqual(calls, ["q"])
        self.assertEqual(result["source_membership"], 1)
        self.assertEqual(result["recall_at_k"], 1)

    def test_unknown_source_is_not_in_context(self):
        self.assertEqual(calculate_source_membership(["fake", "real", "real"], ["real"]), .5)

    def test_missing_sources_get_no_membership_credit(self):
        self.assertEqual(calculate_source_membership([], ["real"]), 0)

    def test_source_presence_cannot_hide_poor_retrieval(self):
        self.assertEqual(quality_gate([{"recall_at_k": .1, "mrr": 1, "source_presence": 1}], .5), 1)

    def test_every_configuration_must_pass(self):
        self.assertEqual(quality_gate([{"recall_at_k": 1, "mrr": 1}, {"recall_at_k": 1, "mrr": .2}], .5), 1)

    def test_nonfinite_score_fails_closed(self):
        self.assertEqual(quality_gate([{"recall_at_k": float("nan"), "mrr": 1}], .5), 1)

    def test_valid_scores_pass(self):
        self.assertEqual(quality_gate([{"recall_at_k": .5, "mrr": .7}], .5), 0)

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            run([], 5, True, False)
