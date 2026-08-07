# WHY BM25 alongside dense embeddings? Embeddings are bad at exact tokens —
# identifiers (ChatGoogleGenerativeAI), error codes (ECONNRESET), version
# strings. BM25 matches literal terms, so the two fail in complementary ways,
# which is exactly what makes fusing them worthwhile.

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_SPLIT = re.compile(r"\W+")


def _tokenise(text: str) -> list[str]:
    """Tokenisation MUST be identical for corpus and query.

    A mismatch between index-time and query-time tokenisation silently
    destroys recall — a classic, hard-to-spot search bug.
    """
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


class BM25Retriever:
    def __init__(self):
        self.corpus: list[str] = []
        self.metadatas: list[dict] = []
        self.ids: list[str] = []
        self.bm25 = None

    def build_index(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ):
        """Tokenise and index the corpus."""
        self.corpus = documents
        self.metadatas = metadatas
        # IDs let the fusion step key on a stable identifier instead of the
        # document text (which is huge, and collides for duplicate content).
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(documents))]
        self.bm25 = BM25Okapi([_tokenise(doc) for doc in documents]) if documents else None

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return up to k hits as {id, content, metadata, score}, best first."""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenise(query))
        # argsort is ascending; [-k:][::-1] takes the top-k in descending order.
        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            {
                "id": self.ids[i],
                "content": self.corpus[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            # Drop documents sharing zero query terms rather than padding with junk.
            for i in top_idx
            if scores[i] > 0
        ]
