"""Base interface for rerankers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Common interface implemented by rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank candidate documents for one query."""
        pass
