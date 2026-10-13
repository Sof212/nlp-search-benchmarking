"""TF-IDF retrieval with scikit-learn cosine similarity."""

from __future__ import annotations

from typing import Any

from src.retrievers.base import BaseRetriever
from src.utils import sorted_results


class TfidfRetriever(BaseRetriever):
    """Sparse lexical retriever based on TF-IDF vectors."""

    def __init__(self, **vectorizer_kwargs: Any) -> None:
        self.vectorizer_kwargs = vectorizer_kwargs
        self.documents: list[dict[str, Any]] = []
        self._vectorizer: Any | None = None
        self._document_matrix: Any | None = None

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Fit a TF-IDF vectorizer over documents."""
        if not documents:
            raise ValueError("TfidfRetriever.index requires at least one document")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError("Install scikit-learn to use TfidfRetriever") from exc

        self.documents = list(documents)
        texts = [str(document["text"]) for document in self.documents]
        self._vectorizer = TfidfVectorizer(**self.vectorizer_kwargs)
        self._document_matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search documents by cosine similarity to a query TF-IDF vector."""
        if self._vectorizer is None or self._document_matrix is None:
            raise RuntimeError("TfidfRetriever must be indexed before search")
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise ImportError("Install scikit-learn to use TfidfRetriever") from exc

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._document_matrix).ravel()
        results = [
            {
                "doc_id": str(document["doc_id"]),
                "score": float(scores[index]),
                "text": str(document["text"]),
            }
            for index, document in enumerate(self.documents)
        ]
        return sorted_results(results, top_k=top_k)
