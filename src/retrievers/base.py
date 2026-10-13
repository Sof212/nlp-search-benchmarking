"""Base interface for retrieval systems."""

from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Common interface implemented by all retrievers."""

    @abstractmethod
    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build an index over benchmark documents."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return top ranked documents for a query."""
        pass

    def batch_search(
        self, queries: list[dict[str, Any]], top_k: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """Search a batch of query records."""
        return {
            query["query_id"]: self.search(query["text"], top_k=top_k)
            for query in queries
        }
