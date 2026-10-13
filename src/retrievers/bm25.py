"""BM25 retrieval using rank-bm25."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.retrievers.base import BaseRetriever
from src.utils import sorted_results


def simple_tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer used by the baseline BM25 retriever."""
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """Lexical BM25 retriever with simple configurable tokenization."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or simple_tokenize
        self.documents: list[dict[str, Any]] = []
        self._bm25: Any | None = None

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build a BM25 index over tokenized documents."""
        if not documents:
            raise ValueError("BM25Retriever.index requires at least one document")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank-bm25 to use BM25Retriever") from exc

        self.documents = list(documents)
        tokenized_corpus = [
            self.tokenizer(str(document["text"])) for document in self.documents
        ]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search documents by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever must be indexed before search")
        query_tokens = self.tokenizer(query)
        scores = self._bm25.get_scores(query_tokens)
        results = [
            {
                "doc_id": str(document["doc_id"]),
                "score": float(scores[index]),
                "text": str(document["text"]),
            }
            for index, document in enumerate(self.documents)
        ]
        return sorted_results(results, top_k=top_k)
