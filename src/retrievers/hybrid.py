"""Hybrid retrieval that fuses multiple retriever outputs."""

from __future__ import annotations

from typing import Any

from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense_embedding import DenseEmbeddingRetriever
from src.utils import normalize_scores, sorted_results


class HybridRetriever(BaseRetriever):
    """Fuse BM25 and dense retrieval with weighted scores or RRF."""

    def __init__(
        self,
        sparse_retriever: BaseRetriever | None = None,
        dense_retriever: BaseRetriever | None = None,
        bm25_config: dict[str, Any] | None = None,
        dense_config: dict[str, Any] | None = None,
        method: str = "rrf",
        alpha: float = 0.5,
        beta: float = 0.5,
        rrf_k: int = 60,
        candidate_pool_size: int = 100,
    ) -> None:
        if method not in {"rrf", "weighted"}:
            raise ValueError("Hybrid method must be either 'rrf' or 'weighted'")
        self.sparse_retriever = sparse_retriever or BM25Retriever(**(bm25_config or {}))
        self.dense_retriever = dense_retriever or DenseEmbeddingRetriever(
            **(dense_config or {})
        )
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Index all component retrievers."""
        if not documents:
            raise ValueError("HybridRetriever.index requires at least one document")
        self.sparse_retriever.index(documents)
        self.dense_retriever.index(documents)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search with component retrievers and fuse the ranked lists."""
        candidate_k = max(top_k, self.candidate_pool_size)
        sparse_results = self.sparse_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        if self.method == "weighted":
            fused = self._weighted_score_fusion(sparse_results, dense_results)
        else:
            fused = self._reciprocal_rank_fusion([sparse_results, dense_results])
        return sorted_results(fused.values(), top_k=top_k)

    def _weighted_score_fusion(
        self,
        sparse_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        sparse_scores = normalize_scores(sparse_results)
        dense_scores = normalize_scores(dense_results)
        documents = _documents_from_results([sparse_results, dense_results])
        fused: dict[str, dict[str, Any]] = {}
        for doc_id, document in documents.items():
            score = self.alpha * sparse_scores.get(doc_id, 0.0) + self.beta * dense_scores.get(
                doc_id, 0.0
            )
            fused[doc_id] = {
                "doc_id": doc_id,
                "score": float(score),
                "text": str(document.get("text", "")),
            }
        return fused

    def _reciprocal_rank_fusion(
        self, ranked_lists: list[list[dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        documents = _documents_from_results(ranked_lists)
        fused: dict[str, dict[str, Any]] = {
            doc_id: {"doc_id": doc_id, "score": 0.0, "text": str(document.get("text", ""))}
            for doc_id, document in documents.items()
        }
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                doc_id = str(result["doc_id"])
                fused[doc_id]["score"] += 1.0 / (self.rrf_k + rank)
        return fused


def _documents_from_results(
    ranked_lists: list[list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for result in ranked_list:
            documents[str(result["doc_id"])] = result
    return documents
