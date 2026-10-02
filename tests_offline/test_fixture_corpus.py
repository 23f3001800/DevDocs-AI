import unittest

from evals.prepare_fixture_corpus import extract_sources


class CorpusBoundaryTests(unittest.TestCase):
    def test_answer_key_is_excluded_and_identity_is_not_indexed(self):
        corpus = "3. Mini Technical Corpus\n" + "\n".join(
            f"doc_example_{i:02} docs/{i}.md\nDocument {i}\nSource text."
            for i in range(1, 14)
        )
        docs = extract_sources([corpus, "4. Test Case Matrix\nTC-01\nExpected docs: secret"])
        self.assertEqual(len(docs), 13)
        self.assertTrue(all("TC-01" not in d["content"] and "doc_example" not in d["content"] for d in docs))

    def test_missing_boundary_fails_closed(self):
        with self.assertRaises(ValueError):
            extract_sources(["3. Mini Technical Corpus\nno end"])
