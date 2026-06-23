# WHY BM25? BM25 uses term frequency + inverse document frequency —
# the same algorithm that powered Google before neural search.
# BM25 catches what dense embeddings miss.

import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.corpus: list[str] = []
        self.metadatas: list[dict] = []
        self.bm25 = None

    def build_index(self, documents: list[str], metadatas: list[dict]):
        """Tokenise and index the corpus."""
        self.corpus = documents
        self.metadatas = metadatas
        tokenised = [re.split(r"\W+", doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenised)

    def search(self, query: str, k: int = 10) -> list[tuple[str, dict, float]]:
        """Return top-k (doc, metadata, score) tuples."""
        if not self.bm25:
            return []
        tokens = re.split(r"\W+", query.lower())
        scores = self.bm25.get_scores(tokens)
        # argsort gives ascending; [-k:][::-1] gives top-k descending
        import numpy as np

        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            (self.corpus[i], self.metadatas[i], float(scores[i])) for i in top_idx if scores[i] > 0
        ]
