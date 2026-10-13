"""Adaptive SPLADE, BM25, dense hybrid retrieval with optional reranking."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.rerankers.base import BaseReranker
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense_embedding import DenseEmbeddingRetriever
from src.retrievers.splade import SpladeRetriever
from src.utils import sorted_results

DEFAULT_WEIGHTS = {"splade": 0.60, "bm25": 0.15, "dense": 0.25}
DEFAULT_ADAPTIVE_WEIGHTS = {
    "entity_heavy": {"splade": 0.65, "bm25": 0.25, "dense": 0.10},
    "long_semantic": {"splade": 0.50, "bm25": 0.10, "dense": 0.40},
    "default": DEFAULT_WEIGHTS,
}
QUESTION_STARTERS = {"what", "why", "how", "whether", "does", "do", "is", "are"}
BIOMEDICAL_TERMS = {
    "biomedical",
    "cancer",
    "cell",
    "clinical",
    "covid",
    "covid-19",
    "disease",
    "dna",
    "drug",
    "enzyme",
    "gene",
    "genomic",
    "kinase",
    "mutation",
    "patient",
    "patients",
    "protein",
    "receptor",
    "rna",
    "trial",
    "vaccine",
}


def infer_query_type(query: str) -> str:
    """Infer a coarse query type using deterministic lexical heuristics."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)
    if not tokens:
        return "default"

    first = tokens[0].lower()
    lower_tokens = {token.lower() for token in tokens}
    question_like = first in QUESTION_STARTERS or query.strip().endswith("?")
    if len(tokens) > 10 or (question_like and len(tokens) > 6):
        return "long_semantic"

    has_number = any(any(char.isdigit() for char in token) for token in tokens)
    has_acronym = any(
        len(token) >= 2 and any(char.isalpha() for char in token) and token.upper() == token
        for token in tokens
    )
    has_capitalized = any(
        token[0].isupper() and token.lower() != first for token in tokens[1:]
    )
    has_biomedical = bool(lower_tokens & BIOMEDICAL_TERMS)
    is_very_short = len(tokens) <= 3
    is_short_non_question = len(tokens) <= 6 and not question_like
    if (
        has_number
        or has_acronym
        or has_capitalized
        or has_biomedical
        or is_very_short
        or (is_short_non_question and has_biomedical)
    ):
        return "entity_heavy"
    return "default"


class AdaptiveSpladeHybridRetriever(BaseRetriever):
    """Fuse SPLADE, BM25, and dense retrieval with adaptive weighted RRF."""

    def __init__(
        self,
        splade_config: dict[str, Any] | None = None,
        bm25_config: dict[str, Any] | None = None,
        dense_config: dict[str, Any] | None = None,
        retrieval_config: dict[str, Any] | None = None,
        fusion_config: dict[str, Any] | None = None,
        rerank_config: dict[str, Any] | None = None,
        splade_retriever: BaseRetriever | None = None,
        bm25_retriever: BaseRetriever | None = None,
        dense_retriever: BaseRetriever | None = None,
        reranker: BaseReranker | None = None,
        device: str | None = None,
    ) -> None:
        self.splade_config = deepcopy(splade_config or {})
        self.bm25_config = deepcopy(bm25_config or {})
        self.dense_config = deepcopy(dense_config or {})
        if device is not None:
            self.splade_config["device"] = device
            self.dense_config["device"] = device

        retrieval = retrieval_config or {}
        self.splade_top_k = int(retrieval.get("splade_top_k", 200))
        self.bm25_top_k = int(retrieval.get("bm25_top_k", 200))
        self.dense_top_k = int(retrieval.get("dense_top_k", 200))
        self.candidate_k = int(retrieval.get("candidate_k", 100))

        fusion = fusion_config or {}
        self.fusion_strategy = str(fusion.get("strategy", "weighted_rrf"))
        if self.fusion_strategy != "weighted_rrf":
            raise ValueError("Only weighted_rrf fusion is currently implemented")
        self.rrf_k = int(fusion.get("rrf_k", 60))
        self.adaptive = bool(fusion.get("adaptive", True))
        self.weights = _merge_weights(DEFAULT_WEIGHTS, fusion.get("weights"))
        self.adaptive_weights = _merge_adaptive_weights(
            DEFAULT_ADAPTIVE_WEIGHTS,
            fusion.get("adaptive_weights"),
        )

        rerank = rerank_config or {}
        self.rerank_enabled = bool(rerank.get("enabled", True))
        self.rerank_model_name = str(
            rerank.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.rerank_top_k = int(rerank.get("top_k", 10))
        self.rerank_batch_size = int(rerank.get("batch_size", 32))
        self.rerank_device = rerank.get("device", device)

        self.splade_retriever = splade_retriever or SpladeRetriever(
            **self.splade_config
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever(**self.bm25_config)
        self.dense_retriever = dense_retriever or DenseEmbeddingRetriever(
            **self.dense_config
        )
        self._reranker = reranker
        self.documents: list[dict[str, Any]] = []

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        splade_retriever: BaseRetriever | None = None,
        bm25_retriever: BaseRetriever | None = None,
        dense_retriever: BaseRetriever | None = None,
        reranker: BaseReranker | None = None,
        device: str | None = None,
    ) -> "AdaptiveSpladeHybridRetriever":
        """Create a retriever from the project YAML config structure."""
        return cls(
            splade_config=dict(config.get("splade", {})),
            bm25_config=dict(config.get("bm25", {})),
            dense_config=dict(config.get("dense", {})),
            retrieval_config=dict(config.get("retrieval", {})),
            fusion_config=dict(config.get("fusion", {})),
            rerank_config=dict(config.get("rerank", {})),
            splade_retriever=splade_retriever,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            reranker=reranker,
            device=device,
        )

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Index all component retrievers."""
        if not documents:
            raise ValueError("AdaptiveSpladeHybridRetriever.index requires documents")
        self.documents = list(documents)
        self.splade_retriever.index(documents)
        self.bm25_retriever.index(documents)
        self.dense_retriever.index(documents)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search with adaptive weighted fusion and optional reranking."""
        splade_results = self.splade_retriever.search(query, top_k=self.splade_top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=self.bm25_top_k)
        dense_results = self.dense_retriever.search(query, top_k=self.dense_top_k)

        query_type = infer_query_type(query)
        weights = self._weights_for_query_type(query_type)
        fused = self.weighted_rrf_fusion(
            ranked_lists={
                "splade": splade_results,
                "bm25": bm25_results,
                "dense": dense_results,
            },
            weights=weights,
            rrf_k=self.rrf_k,
            top_k=self.candidate_k,
        )
        if not self.rerank_enabled:
            return sorted_results(fused, top_k=top_k)
        return self._load_reranker().rerank(query, fused, top_k=top_k)

    @staticmethod
    def weighted_rrf_fusion(
        ranked_lists: dict[str, list[dict[str, Any]]],
        weights: dict[str, float],
        rrf_k: int = 60,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Fuse ranked lists with weighted reciprocal rank fusion."""
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        fused: dict[str, dict[str, Any]] = {}
        for source, results in ranked_lists.items():
            weight = float(weights.get(source, 0.0))
            if weight <= 0:
                continue
            for rank, result in enumerate(results, start=1):
                doc_id = str(result["doc_id"])
                entry = fused.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": 0.0,
                        "text": str(result.get("text", "")),
                    },
                )
                if not entry.get("text") and result.get("text"):
                    entry["text"] = str(result["text"])
                entry["score"] += weight * (1.0 / (rrf_k + rank))
        return sorted_results(fused.values(), top_k=top_k)

    def _weights_for_query_type(self, query_type: str) -> dict[str, float]:
        if not self.adaptive:
            return self.weights
        return self.adaptive_weights.get(query_type, self.adaptive_weights["default"])

    def _load_reranker(self) -> BaseReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.rerank_model_name,
                batch_size=self.rerank_batch_size,
                device=self.rerank_device,
            )
        return self._reranker


def _merge_weights(
    defaults: dict[str, float],
    overrides: dict[str, Any] | None,
) -> dict[str, float]:
    merged = dict(defaults)
    if overrides:
        merged.update({key: float(value) for key, value in overrides.items()})
    return merged


def _merge_adaptive_weights(
    defaults: dict[str, dict[str, float]],
    overrides: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    merged = {name: dict(weights) for name, weights in defaults.items()}
    if overrides:
        for query_type, weights in overrides.items():
            if isinstance(weights, dict):
                merged[query_type] = _merge_weights(
                    merged.get(query_type, DEFAULT_WEIGHTS),
                    weights,
                )
    return merged
