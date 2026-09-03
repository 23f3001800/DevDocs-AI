"""Small, dependency-free metrics used by the offline evaluation runner."""

import re
from collections.abc import Iterable, Sequence


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def calculate_recall_at_k(
    relevant_ids: set[str], retrieved_ids: Sequence[str], k: int = 3
) -> float:
    """Return the fraction of relevant documents found in the first *k* results."""
    if not relevant_ids or k <= 0:
        return 0.0
    hits = set(_unique(retrieved_ids[:k])) & relevant_ids
    return len(hits) / len(relevant_ids)


def calculate_precision_at_k(
    relevant_ids: set[str], retrieved_ids: Sequence[str], k: int = 3
) -> float:
    """Return the fraction of unique first-*k* results that are relevant."""
    top_k = _unique(retrieved_ids[:k]) if k > 0 else []
    return sum(doc_id in relevant_ids for doc_id in top_k) / len(top_k) if top_k else 0.0


def calculate_mrr(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int = 5) -> float:
    """Return reciprocal rank of the first relevant result, or zero."""
    if k <= 0:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_hit_rate(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int = 3) -> float:
    """Return one when any relevant result appears in the first *k* results."""
    return float(bool(set(retrieved_ids[:k]) & relevant_ids)) if k > 0 else 0.0


def calculate_keyword_coverage(answer: str, ground_truth: str) -> float:
    """Approximate answer correctness without requiring a paid LLM judge."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "can",
        "do",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "under",
        "within",
        "you",
    }
    expected = {
        word
        for word in re.findall(r"[a-z0-9]+", ground_truth.lower())
        if len(word) > 2 and word not in stop_words
    }
    if not expected:
        return 1.0 if answer.strip() else 0.0
    actual = set(re.findall(r"[a-z0-9]+", answer.lower()))
    return len(expected & actual) / len(expected)


def calculate_citation_coverage(answer: str, sources: Sequence[str]) -> float:
    """Return one when the model provides sources for its answer in the structured output."""
    has_answer = bool(answer.strip())
    has_sources = len(sources) > 0
    if has_answer:
        return 1.0 if has_sources else 0.0
    else:
        return 1.0 if not has_sources else 0.0
