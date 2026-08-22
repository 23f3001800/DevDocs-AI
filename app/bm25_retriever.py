# BM25 matches literal tokens (identifiers, error codes, version strings) that
# dense embeddings miss — the two fail in complementary ways, so we fuse them.

import re

from rank_bm25 import BM25Okapi

_TOKEN_SPLIT = re.compile(r"\W+")


def _tokenise(text: str) -> list[str]:
    """Tokenisation MUST be identical for corpus and query, or recall silently drops."""
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

    def search(self, query: str, k: int = 10, owner: str | None = None) -> list[dict]:
        """Return up to k hits as {id, content, metadata, score}, best first.

        No `score > 0` filter: rank_bm25's IDF can go negative for common terms,
        so a legitimate match can score at/below zero. Relevance is the
        reranker's job, not a sign check here.

        `owner` isolation is applied BEFORE the top-k cut (sparse search has no
        native metadata filter), so filtering never starves the result set.
        Chunks with no `owner` key never match, so legacy chunks stay private.
        """
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenise(query))
        if owner is not None:
            candidate_idx = [i for i, m in enumerate(self.metadatas) if m.get("owner") == owner]
            if not candidate_idx:
                return []
        else:
            candidate_idx = range(len(scores))
        top_idx = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "id": self.ids[i],
                "content": self.corpus[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]
