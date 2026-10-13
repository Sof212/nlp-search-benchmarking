"""Cross-encoder reranking with sentence-transformers."""

from __future__ import annotations

from typing import Any

from src.rerankers.base import BaseReranker
from src.utils import sorted_results


class CrossEncoderReranker(BaseReranker):
    """Rerank query-document pairs with a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model: Any | None = None

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Score candidate documents and return the best ``top_k``."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, str(document.get("text", ""))) for document in documents]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked: list[dict[str, Any]] = []
        for document, score in zip(documents, scores):
            item = dict(document)
            if "score" in item:
                item["retrieval_score"] = float(item["score"])
            item["score"] = float(score)
            reranked.append(item)
        return sorted_results(reranked, top_k=top_k)

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers to use CrossEncoderReranker"
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model
