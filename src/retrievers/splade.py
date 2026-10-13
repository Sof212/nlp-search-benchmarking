"""SPLADE sparse neural retrieval.

The implementation keeps model loading lazy because SPLADE models are large.
It is intentionally modular so indexing can later be swapped for FAISS,
Pyserini, or persisted sparse matrices without changing the retriever API.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.retrievers.base import BaseRetriever
from src.utils import sorted_results


class SpladeRetriever(BaseRetriever):
    """Sparse neural retriever using a masked language model SPLADE head."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        batch_size: int = 8,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.documents: list[dict[str, Any]] = []
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._document_vectors: np.ndarray | None = None

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Encode documents into SPLADE sparse vectors."""
        if not documents:
            raise ValueError("SpladeRetriever.index requires at least one document")
        self.documents = list(documents)
        texts = [str(document["text"]) for document in self.documents]
        self._document_vectors = self.encode(texts)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search by sparse dot product between SPLADE vectors."""
        if self._document_vectors is None:
            raise RuntimeError("SpladeRetriever must be indexed before search")
        query_vector = self.encode([query])[0]
        scores = self._document_vectors @ query_vector
        results = [
            {
                "doc_id": str(document["doc_id"]),
                "score": float(scores[index]),
                "text": str(document["text"]),
            }
            for index, document in enumerate(self.documents)
        ]
        return sorted_results(results, top_k=top_k)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text into SPLADE sparse lexical vectors."""
        tokenizer, model, torch = self._load_model()
        vectors: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                logits = model(**inputs).logits
                attention = inputs["attention_mask"].unsqueeze(-1)
                sparse = torch.log1p(torch.relu(logits)) * attention
                sparse = torch.max(sparse, dim=1).values
                vectors.append(sparse.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Install torch and transformers to use SpladeRetriever"
                ) from exc
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(device)
        else:
            import torch
        return self._tokenizer, self._model, torch
