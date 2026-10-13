"""Dense embedding retrieval with sentence-transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.retrievers.base import BaseRetriever
from src.utils import sorted_results


class DenseEmbeddingRetriever(BaseRetriever):
    """Dense vector retriever using cosine similarity."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.device = device
        self.documents: list[dict[str, Any]] = []
        self._model: Any | None = None
        self._document_embeddings: np.ndarray | None = None

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Encode documents into dense embeddings."""
        if not documents:
            raise ValueError("DenseEmbeddingRetriever.index requires at least one document")
        self.documents = list(documents)
        texts = [
            f"{self.document_prefix}{document['text']}" for document in self.documents
        ]
        model = self._load_model()
        self._document_embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search by cosine similarity against dense document embeddings."""
        if self._document_embeddings is None:
            raise RuntimeError("DenseEmbeddingRetriever must be indexed before search")
        model = self._load_model()
        query_embedding = model.encode(
            [f"{self.query_prefix}{query}"],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        scores = _cosine_scores(query_embedding, self._document_embeddings)
        results = [
            {
                "doc_id": str(document["doc_id"]),
                "score": float(scores[index]),
                "text": str(document["text"]),
            }
            for index, document in enumerate(self.documents)
        ]
        return sorted_results(results, top_k=top_k)

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers to use DenseEmbeddingRetriever"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


def _cosine_scores(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)
    denominator = np.maximum(query_norm * document_norms, 1e-12)
    return documents @ query / denominator
