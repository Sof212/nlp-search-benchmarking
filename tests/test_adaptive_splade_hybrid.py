from __future__ import annotations

from typing import Any

from src.retrievers.adaptive_splade_hybrid import (
    AdaptiveSpladeHybridRetriever,
    infer_query_type,
)
from src.retrievers.base import BaseRetriever


class FakeRetriever(BaseRetriever):
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.indexed = False

    def index(self, documents: list[dict[str, Any]]) -> None:
        self.indexed = True

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.results[:top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.called = False

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        self.called = True
        reranked = list(reversed(documents))
        for result in reranked:
            result["score"] = result["score"] + 1.0
        return reranked[:top_k]


def test_infer_query_type_entity_heavy() -> None:
    assert infer_query_type("BRCA1 TP53 cancer") == "entity_heavy"
    assert infer_query_type("COVID-19 vaccine") == "entity_heavy"


def test_infer_query_type_long_semantic() -> None:
    query = (
        "What evidence explains whether dense semantic retrieval improves "
        "scientific search over lexical matching?"
    )
    assert infer_query_type(query) == "long_semantic"


def test_infer_query_type_default() -> None:
    assert infer_query_type("neural retrieval model comparison") == "default"


def test_weighted_rrf_prefers_high_weight_source() -> None:
    fused = AdaptiveSpladeHybridRetriever.weighted_rrf_fusion(
        ranked_lists={
            "splade": [{"doc_id": "A", "score": 1.0, "text": "alpha"}],
            "bm25": [],
            "dense": [{"doc_id": "B", "score": 1.0, "text": "beta"}],
        },
        weights={"splade": 0.70, "bm25": 0.0, "dense": 0.30},
        rrf_k=60,
        top_k=2,
    )

    assert [result["doc_id"] for result in fused] == ["A", "B"]


def test_fusion_includes_documents_from_all_retrievers() -> None:
    fused = AdaptiveSpladeHybridRetriever.weighted_rrf_fusion(
        ranked_lists={
            "splade": [{"doc_id": "A", "score": 1.0, "text": "alpha"}],
            "bm25": [{"doc_id": "B", "score": 1.0, "text": "beta"}],
            "dense": [{"doc_id": "C", "score": 1.0, "text": "gamma"}],
        },
        weights={"splade": 0.60, "bm25": 0.15, "dense": 0.25},
        rrf_k=60,
        top_k=10,
    )

    assert {result["doc_id"] for result in fused} == {"A", "B", "C"}


def test_search_without_reranking_returns_compatible_output() -> None:
    retriever = AdaptiveSpladeHybridRetriever(
        splade_retriever=FakeRetriever(
            [{"doc_id": "A", "score": 1.0, "text": "alpha"}]
        ),
        bm25_retriever=FakeRetriever(
            [{"doc_id": "B", "score": 1.0, "text": "beta"}]
        ),
        dense_retriever=FakeRetriever(
            [{"doc_id": "C", "score": 1.0, "text": "gamma"}]
        ),
        rerank_config={"enabled": False},
    )

    results = retriever.search("neural retrieval model comparison", top_k=2)

    assert len(results) == 2
    assert {"doc_id", "score", "text"} <= set(results[0])
    assert results[0]["doc_id"] == "A"


def test_reranking_can_be_disabled() -> None:
    reranker = FakeReranker()
    retriever = AdaptiveSpladeHybridRetriever(
        splade_retriever=FakeRetriever(
            [{"doc_id": "A", "score": 1.0, "text": "alpha"}]
        ),
        bm25_retriever=FakeRetriever([]),
        dense_retriever=FakeRetriever([]),
        reranker=reranker,
        rerank_config={"enabled": False},
    )

    results = retriever.search("BRCA1", top_k=1)

    assert not reranker.called
    assert results[0]["doc_id"] == "A"
