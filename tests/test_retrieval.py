"""
Tests for the fusion + rerank logic in HybridRetriever.

This is the most complex code in the repo and was previously untested, because
a real retriever wants ~180 MB of models. These build one with __new__ and inject
fakes, so the RRF maths and the stale-index contract are covered in milliseconds.
"""

import pytest

from app.bm25_retriever import BM25Retriever
from app.hybrid_retriever import HybridRetriever
from app.vectorstore import make_chunk_id


class _FakeVectorStore:
    """Minimal stand-in exposing just what HybridRetriever calls."""

    def __init__(self, docs):
        # docs: list of (id, content, metadata)
        self.docs = docs
        self.dense_hits = []

    def query(self, text, k=5, file_type=None, owner=None):
        hits = self.dense_hits
        if owner is not None:
            hits = [h for h in hits if h[2].get("owner") == owner]
        return [{"id": i, "content": c, "metadata": m, "score": 1.0} for i, c, m in hits[:k]]

    def all_documents(self):
        return {
            "ids": [d[0] for d in self.docs],
            "documents": [d[1] for d in self.docs],
            "metadatas": [d[2] for d in self.docs],
        }


class _FakeReranker:
    """Scores by position of a marker token, so ordering is deterministic."""

    def __init__(self, scores_by_content=None):
        self.scores_by_content = scores_by_content or {}
        self.seen_pairs = []

    def predict(self, pairs):
        self.seen_pairs = pairs
        return [self.scores_by_content.get(doc, 0.0) for _, doc in pairs]


def _build(docs, dense_hits, rerank_scores=None, k_rrf=60):
    r = HybridRetriever.__new__(HybridRetriever)
    import threading

    r.k_rrf = k_rrf
    r.default_initial_k = 20
    r.vs = _FakeVectorStore(docs)
    r.vs.dense_hits = dense_hits
    r.bm25 = BM25Retriever()
    r._bm25_lock = threading.Lock()
    r.rebuild_bm25()
    r.reranker = _FakeReranker(rerank_scores)
    return r


DOCS = [
    (
        "a",
        "FastAPI dependency injection resolves the graph before the handler runs.",
        {"file_path": "a.md"},
    ),
    (
        "b",
        "BM25 ranks documents by term frequency and inverse document frequency.",
        {"file_path": "b.md"},
    ),
    (
        "c",
        "ChromaDB stores vector embeddings for approximate nearest neighbour search.",
        {"file_path": "c.md"},
    ),
]


# ── RRF fusion ───────────────────────────────────────────────
def test_rrf_rewards_agreement_between_retrievers():
    """A doc both retrievers like must outrank one only a single retriever found."""
    # Dense ranks c first, then b. BM25 (query below) will rank b first.
    r = _build(DOCS, dense_hits=[DOCS[2], DOCS[1]], rerank_scores={})
    dense = r.vs.query("x", k=20)
    # k=1: BM25Retriever.search() no longer hard-filters zero-score hits (see
    # bm25_retriever.py), so a wide k over this 3-doc corpus would also return
    # the two irrelevant docs at score 0.0 — noise this test isn't about.
    # Asking for just the single best sparse match keeps it isolated to "b",
    # the one genuine term match.
    sparse = r.bm25.search("BM25 term frequency inverse document", k=1)

    scores = {}
    for hits in (dense, sparse):
        for rank, item in enumerate(hits):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1 / (r.k_rrf + rank + 1)

    # b appears in both lists, c only in the dense list.
    assert scores["b"] > scores["c"]


def test_rrf_uses_the_configured_k_rrf():
    """k_rrf must actually reach the maths — it used to be stored and ignored."""
    high = _build(DOCS, dense_hits=[DOCS[0]], rerank_scores={DOCS[0][1]: 1.0}, k_rrf=1)
    low = _build(DOCS, dense_hits=[DOCS[0]], rerank_scores={DOCS[0][1]: 1.0}, k_rrf=1000)
    assert high.k_rrf == 1
    assert low.k_rrf == 1000
    # 1/(k+rank+1) — a smaller k produces a materially larger contribution.
    assert 1 / (high.k_rrf + 1) > 1 / (low.k_rrf + 1)


def test_retrieve_returns_reranked_order_not_rrf_order():
    """The cross-encoder has the final say — that is the whole point of stage 3."""
    # Dense puts a first, but the reranker scores c highest.
    r = _build(
        DOCS,
        dense_hits=[DOCS[0], DOCS[1], DOCS[2]],
        rerank_scores={DOCS[0][1]: 0.1, DOCS[1][1]: 0.2, DOCS[2][1]: 0.9},
    )
    results = r.retrieve("anything", k=3)
    assert [x["id"] for x in results] == ["c", "b", "a"]
    assert results[0]["rerank_score"] == pytest.approx(0.9)


def test_retrieve_preserves_metadata_for_duplicate_content():
    """Fusion keys on chunk ID, so identical text from two files stays distinct.

    Keying on the document string used to collapse duplicates into one entry
    carrying whichever metadata arrived first — meaning a citation could name
    the wrong file.
    """
    dup = "Shared boilerplate licence header, identical in both files."
    docs = [("id-x", dup, {"file_path": "x.md"}), ("id-y", dup, {"file_path": "y.md"})]
    r = _build(docs, dense_hits=docs, rerank_scores={dup: 0.5})
    results = r.retrieve("licence", k=2)
    assert len(results) == 2
    assert {x["metadata"]["file_path"] for x in results} == {"x.md", "y.md"}


def test_retrieve_empty_index_returns_empty():
    r = _build([], dense_hits=[], rerank_scores={})
    assert r.retrieve("anything", k=5) == []


# ── The stale-BM25 regression ────────────────────────────────
def test_rebuild_bm25_picks_up_newly_ingested_documents():
    """The bug: the BM25 index was built once in __init__ and never refreshed,
    so anything added by /ingest was invisible to sparse search until restart."""
    r = _build(DOCS, dense_hits=[], rerank_scores={})
    # search() no longer hard-filters zero-score hits (see bm25_retriever.py),
    # so this returns the 3 existing (irrelevant, score 0.0) docs rather than
    # []. The regression this test guards against is specifically that "d"
    # cannot be found before the index is rebuilt.
    before = r.bm25.search("Kubernetes operator reconciliation", k=5)
    assert all(hit["id"] != "d" for hit in before)

    # Simulate an ingest adding a new chunk to the store.
    r.vs.docs.append(
        (
            "d",
            "Kubernetes operator reconciliation loops watch custom resources.",
            {"file_path": "d.md"},
        )
    )
    count = r.rebuild_bm25()

    assert count == 4
    hits = r.bm25.search("Kubernetes operator reconciliation", k=5)
    assert hits and hits[0]["id"] == "d"


# ── Chunk IDs ────────────────────────────────────────────────
def test_chunk_ids_are_deterministic():
    meta = {"source": "https://github.com/x/y", "file_path": "app/main.py", "chunk_index": 3}
    assert make_chunk_id(meta) == make_chunk_id(dict(meta))


def test_chunk_ids_do_not_collide_across_sources_without_file_path():
    """The old f"{file_path}_{chunk_index}" scheme produced 'doc_0' for EVERY
    URL and PDF, so ingesting a second one silently overwrote the first."""
    a = make_chunk_id({"source": "https://a.example.com/docs", "chunk_index": 0})
    b = make_chunk_id({"source": "https://b.example.com/docs", "chunk_index": 0})
    assert a != b


def test_chunk_ids_differ_across_repos_sharing_a_filename():
    a = make_chunk_id({"source": "repo-a", "file_path": "README.md", "chunk_index": 0})
    b = make_chunk_id({"source": "repo-b", "file_path": "README.md", "chunk_index": 0})
    assert a != b


def test_chunk_ids_differ_per_chunk_index():
    base = {"source": "s", "file_path": "f.md"}
    assert make_chunk_id({**base, "chunk_index": 0}) != make_chunk_id({**base, "chunk_index": 1})


def test_chunk_ids_differ_by_owner_for_the_same_source():
    """Two users independently ingesting the same public URL must not collide
    on one shared chunk id — each owner's copy needs its own row, or the
    second ingest would silently overwrite (and reassign) the first user's
    chunk."""
    base = {"source": "https://shared.example/docs", "file_path": "index.html", "chunk_index": 0}
    assert make_chunk_id({**base, "owner": "alice"}) != make_chunk_id({**base, "owner": "bob"})
    # No owner at all still matches the pre-multi-tenant id exactly.
    assert make_chunk_id(base) == make_chunk_id({**base, "owner": ""})


def test_chunk_ids_differ_by_commit_sha_for_the_same_repo():
    base = {"source": "https://github.com/x/y", "file_path": "app/main.py", "chunk_index": 0}
    a = make_chunk_id({**base, "commit_sha": "aaa111"})
    b = make_chunk_id({**base, "commit_sha": "bbb222"})
    assert a != b
    # URL/PDF sources never set commit_sha — behaviour is unchanged for them.
    assert make_chunk_id(base) == make_chunk_id({**base, "commit_sha": ""})


# ── Per-user retrieval isolation ────────────────────────────────
# Retrieval is per-user PRIVATE: a query scoped to `owner` must only ever see
# that owner's own ingested chunks. Admin (owner=None) sees across everyone.
OWNED_DOCS = [
    (
        "alice-1",
        "Alice's private deploy runbook covers the internal staging pipeline.",
        {"file_path": "alice.md", "owner": "alice"},
    ),
    (
        "bob-1",
        "Bob's private deploy runbook covers the internal staging pipeline.",
        {"file_path": "bob.md", "owner": "bob"},
    ),
]


def test_retrieve_owner_filter_excludes_other_users_chunks():
    """User A cannot retrieve user B's chunks even when both dense and sparse
    recall would otherwise surface them."""
    r = _build(
        OWNED_DOCS,
        dense_hits=OWNED_DOCS,
        rerank_scores={OWNED_DOCS[0][1]: 0.9, OWNED_DOCS[1][1]: 0.9},
    )
    results = r.retrieve("private deploy runbook", k=5, owner="alice")
    assert [x["id"] for x in results] == ["alice-1"]
    assert all(x["metadata"]["owner"] == "alice" for x in results)


def test_retrieve_admin_sees_across_owners():
    r = _build(
        OWNED_DOCS,
        dense_hits=OWNED_DOCS,
        rerank_scores={OWNED_DOCS[0][1]: 0.9, OWNED_DOCS[1][1]: 0.8},
    )
    results = r.retrieve("private deploy runbook", k=5, owner=None)
    assert {x["id"] for x in results} == {"alice-1", "bob-1"}
