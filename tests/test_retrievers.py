from __future__ import annotations

import pytest

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.tfidf import TfidfRetriever


DOCUMENTS = [
    {"doc_id": "d1", "text": "apple banana fruit"},
    {"doc_id": "d2", "text": "engine wheel vehicle"},
    {"doc_id": "d3", "text": "ranking function search retrieval"},
]


def test_tfidf_retriever_returns_expected_document() -> None:
    pytest.importorskip("sklearn")
    retriever = TfidfRetriever()
    retriever.index(DOCUMENTS)

    results = retriever.search("apple fruit", top_k=2)

    assert len(results) == 2
    assert results[0]["doc_id"] == "d1"
    assert {"doc_id", "score", "text"} <= set(results[0])


def test_bm25_retriever_returns_expected_document() -> None:
    pytest.importorskip("rank_bm25")
    retriever = BM25Retriever()
    retriever.index(DOCUMENTS)

    results = retriever.search("ranking function", top_k=2)

    assert len(results) == 2
    assert results[0]["doc_id"] == "d3"
    assert {"doc_id", "score", "text"} <= set(results[0])
