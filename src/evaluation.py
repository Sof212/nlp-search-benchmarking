"""Evaluation metrics for ranked retrieval runs."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

QrelsForQuery = dict[str, int]
Run = dict[str, list[dict[str, Any]]]
Qrels = dict[str, QrelsForQuery]


def precision_at_k(qrels: QrelsForQuery, results: list[dict[str, Any]], k: int) -> float:
    """Compute Precision@K for one query."""
    _validate_k(k)
    if not results:
        return 0.0
    relevant_retrieved = sum(
        1 for result in results[:k] if qrels.get(str(result.get("doc_id")), 0) > 0
    )
    return relevant_retrieved / k


def recall_at_k(qrels: QrelsForQuery, results: list[dict[str, Any]], k: int) -> float:
    """Compute Recall@K for one query."""
    _validate_k(k)
    relevant_docs = {doc_id for doc_id, relevance in qrels.items() if relevance > 0}
    if not relevant_docs:
        return 0.0
    retrieved = {str(result.get("doc_id")) for result in results[:k]}
    return len(relevant_docs & retrieved) / len(relevant_docs)


def mrr_at_k(qrels: QrelsForQuery, results: list[dict[str, Any]], k: int) -> float:
    """Compute reciprocal rank of the first relevant result within K."""
    _validate_k(k)
    for rank, result in enumerate(results[:k], start=1):
        if qrels.get(str(result.get("doc_id")), 0) > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(qrels: QrelsForQuery, results: list[dict[str, Any]], k: int) -> float:
    """Compute Discounted Cumulative Gain@K using graded relevance."""
    _validate_k(k)
    score = 0.0
    for rank, result in enumerate(results[:k], start=1):
        relevance = qrels.get(str(result.get("doc_id")), 0)
        if relevance > 0:
            score += (2**relevance - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(qrels: QrelsForQuery, results: list[dict[str, Any]], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain@K."""
    _validate_k(k)
    ideal_relevances = sorted(qrels.values(), reverse=True)[:k]
    ideal_results = [
        {"doc_id": f"__ideal_{index}__"}
        for index, relevance in enumerate(ideal_relevances)
    ]
    ideal_qrels = {
        f"__ideal_{index}__": relevance
        for index, relevance in enumerate(ideal_relevances)
    }
    ideal_dcg = dcg_at_k(ideal_qrels, ideal_results, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(qrels, results, k) / ideal_dcg


def evaluate_run(
    qrels: Qrels,
    run: Run,
    k_values: list[int] | tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Evaluate a complete run across all qrels queries.

    Args:
        qrels: Nested mapping of query IDs to document relevance labels.
        run: Mapping of query IDs to ranked retrieval results.
        k_values: Cutoffs to evaluate.

    Returns:
        Metric names mapped to macro-averaged scores.
    """
    if not qrels:
        raise ValueError("Cannot evaluate an empty qrels mapping")

    cutoffs = sorted(set(k_values or [10, 100]))
    for cutoff in cutoffs:
        _validate_k(cutoff)

    query_ids = sorted(qrels)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        per_query_results = [
            (
                precision_at_k(qrels[query_id], run.get(query_id, []), cutoff),
                recall_at_k(qrels[query_id], run.get(query_id, []), cutoff),
                mrr_at_k(qrels[query_id], run.get(query_id, []), cutoff),
                ndcg_at_k(qrels[query_id], run.get(query_id, []), cutoff),
            )
            for query_id in query_ids
        ]
        precision_scores, recall_scores, mrr_scores, ndcg_scores = zip(
            *per_query_results
        )
        metrics[f"Precision@{cutoff}"] = mean(precision_scores)
        metrics[f"Recall@{cutoff}"] = mean(recall_scores)
        metrics[f"MRR@{cutoff}"] = mean(mrr_scores)
        metrics[f"NDCG@{cutoff}"] = mean(ndcg_scores)
    return metrics


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
