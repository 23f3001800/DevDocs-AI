"""Offline retrieval ablation for chunk size and overlap choices.

The evaluation PDF contains both a mini corpus and its answer key. This runner
extracts only the corpus section, preventing ground-truth leakage, then builds
small in-memory hybrid indexes for each configuration. It never modifies the
production Chroma collection.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.bm25_retriever import BM25Retriever
from app.retriever_instance import get_retriever
from evals.run_evals import load_cases, run

ROOT = Path(__file__).resolve().parents[1]
DOC_HEADER = re.compile(r"(?m)^(doc_[a-z]+_\d{2}) ([^\n]+)\n")


def load_fixture_corpus(path: Path) -> list[tuple[str, str, str]]:
    """Return ``(doc_id, source_path, text)`` for the 13 corpus documents."""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    start = text.index("doc_arch_01")
    end = text.index("4. Test Matrix", start)
    corpus = text[start:end]
    matches = list(DOC_HEADER.finditer(corpus))
    documents = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(corpus)
        body = corpus[match.end() : body_end]
        # PDF page furniture and bullet artifacts are not source content.
        body = re.sub(r"(?m)^Page \d+$", "", body)
        body = body.replace("•", " ")
        documents.append((match.group(1), match.group(2), body.strip()))
    if len(documents) != 13:
        raise ValueError(f"Expected 13 fixture documents, found {len(documents)}")
    return documents


class InMemoryHybridRetriever:
    """Dense + BM25 + RRF + cross-encoder over an isolated chunk variant."""

    def __init__(
        self,
        documents,
        chunk_size: int,
        overlap: int,
        initial_k: int = 10,
        mode: str = "rerank",
    ):
        shared = get_retriever()
        self.embedder = shared.vs.embedder
        self.reranker = shared.reranker
        self.initial_k = initial_k
        self.rrf_k = shared.k_rrf
        self.mode = mode

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = []
        for doc_id, source_path, body in documents:
            for index, chunk in enumerate(splitter.split_text(body)):
                self.chunks.append(
                    {
                        "id": f"{doc_id}:{index}",
                        # Keep doc_id in-band because the standard evaluator
                        # maps retrieved chunks back to ground-truth IDs.
                        "content": f"{doc_id}\n{chunk}",
                        "metadata": {"doc_id": doc_id, "file_path": source_path},
                    }
                )

        texts = [chunk["content"] for chunk in self.chunks]
        self.embeddings = np.asarray(
            self.embedder.encode(texts, normalize_embeddings=True), dtype=np.float32
        )
        self.bm25 = BM25Retriever()
        self.bm25.build_index(
            texts,
            [chunk["metadata"] for chunk in self.chunks],
            [chunk["id"] for chunk in self.chunks],
        )

    def retrieve(self, query: str, k: int = 5, owner=None) -> list[dict]:
        query_embedding = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True)[0], dtype=np.float32
        )
        dense_scores = self.embeddings @ query_embedding
        dense_indices = np.argsort(-dense_scores)[: self.initial_k]
        dense = [
            {**self.chunks[index], "score": float(dense_scores[index])}
            for index in dense_indices
        ]
        sparse = self.bm25.search(query, k=self.initial_k)

        if self.mode == "dense":
            return [{**hit, "rerank_score": hit["score"]} for hit in dense[:k]]
        if self.mode == "bm25":
            return [{**hit, "rerank_score": hit["score"]} for hit in sparse[:k]]

        by_id = {chunk["id"]: chunk for chunk in self.chunks}
        rrf_scores: dict[str, float] = {}
        for hits in (dense, sparse):
            for rank, hit in enumerate(hits):
                doc_id = hit["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    self.rrf_k + rank + 1
                )

        candidates = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        if self.mode == "hybrid":
            return [
                {**by_id[doc_id], "rerank_score": rrf_scores[doc_id]}
                for doc_id in candidates[:k]
            ]
        pairs = [[query, by_id[doc_id]["content"]] for doc_id in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:k]
        return [
            {
                **by_id[doc_id],
                "rerank_score": float(score),
            }
            for doc_id, score in ranked
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", type=Path, default=ROOT / "data" / "devdocs_ragas_eval_test_cases.pdf"
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "eval_dataset.json")
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Also call the configured LLM for the clean four-strategy comparison",
    )
    args = parser.parse_args()

    documents = load_fixture_corpus(args.pdf)
    cases = load_cases(args.dataset)
    results = {"strategy_comparison": {}, "k_ablation": {}, "chunk_ablation": {}}
    for label, mode in (
        ("Dense", "dense"),
        ("BM25", "bm25"),
        ("Hybrid", "hybrid"),
        ("Hybrid + Reranking", "rerank"),
    ):
        retriever = InMemoryHybridRetriever(documents, 600, 60, mode=mode)
        results["strategy_comparison"][label] = run(
            cases,
            k=5,
            retrieval_only=not args.generation,
            use_ragas=False,
            retriever=retriever,
        )

    for initial_k, final_k in ((10, 3), (10, 5), (20, 5), (20, 8), (40, 5)):
        name = f"candidate_{initial_k}_top_{final_k}"
        retriever = InMemoryHybridRetriever(
            documents,
            600,
            60,
            initial_k=initial_k,
            mode="rerank",
        )
        results["k_ablation"][name] = run(
            cases,
            k=final_k,
            retrieval_only=True,
            use_ragas=False,
            retriever=retriever,
        )

    variants = [(400, 0), (400, 40), (600, 0), (600, 60), (800, 80)]
    for chunk_size, overlap in variants:
        name = f"chunk_{chunk_size}_overlap_{overlap}"
        retriever = InMemoryHybridRetriever(documents, chunk_size, overlap)
        results["chunk_ablation"][name] = run(
            cases,
            k=5,
            retrieval_only=True,
            use_ragas=False,
            retriever=retriever,
        )
        results["chunk_ablation"][name]["chunks"] = len(retriever.chunks)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
