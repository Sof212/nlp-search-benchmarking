from __future__ import annotations

import pytest

from src.evaluation import (
    dcg_at_k,
    evaluate_run,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_pointwise_metrics() -> None:
    qrels = {"d1": 1, "d2": 0, "d3": 2}
    results = [
        {"doc_id": "d3", "score": 3.0},
        {"doc_id": "d9", "score": 2.0},
        {"doc_id": "d1", "score": 1.0},
    ]

    assert precision_at_k(qrels, results, 2) == 0.5
    assert recall_at_k(qrels, results, 2) == 0.5
    assert mrr_at_k(qrels, results, 3) == 1.0
    assert dcg_at_k(qrels, results, 1) == 3.0
    assert ndcg_at_k(qrels, results, 1) == 1.0


def test_evaluate_run_macro_averages() -> None:
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d3": 1},
    }
    run = {
        "q1": [{"doc_id": "d1", "score": 2.0}],
        "q2": [
            {"doc_id": "d2", "score": 3.0},
            {"doc_id": "d3", "score": 2.0},
        ],
    }

    metrics = evaluate_run(qrels, run, k_values=[1, 2])

    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@2"] == 1.0
    assert metrics["MRR@2"] == 0.75
    assert metrics["Precision@1"] == 0.5


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k({}, [], 0)
