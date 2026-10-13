from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.dataset import load_ir_dataset, validate_ir_dataset_export_request


class FakeIrDataset:
    def docs_iter(self) -> Any:
        return iter(
            [
                SimpleNamespace(doc_id="d1", title="Title 1", text="Document one."),
                SimpleNamespace(doc_id="d2", text="Document two."),
                SimpleNamespace(doc_id="d3", text="Document three."),
            ]
        )

    def queries_iter(self) -> Any:
        return iter(
            [
                SimpleNamespace(query_id="q1", text="first query"),
                SimpleNamespace(query_id="q2", text="second query"),
                SimpleNamespace(query_id="q3", text="third query"),
            ]
        )

    def qrels_iter(self) -> Any:
        return iter(
            [
                SimpleNamespace(query_id="q1", doc_id="d1", relevance=2),
                SimpleNamespace(query_id="q2", doc_id="d2", relevance=1),
                SimpleNamespace(query_id="q3", doc_id="d3", relevance=1),
            ]
        )


def install_fake_ir_datasets(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loaded_names: list[str] = []

    def load(dataset_name: str) -> FakeIrDataset:
        loaded_names.append(dataset_name)
        return FakeIrDataset()

    monkeypatch.setitem(sys.modules, "ir_datasets", SimpleNamespace(load=load))
    return loaded_names


def test_load_ir_dataset_converts_docs_queries_and_qrels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_names = install_fake_ir_datasets(monkeypatch)

    documents, queries, qrels = load_ir_dataset("beir/scifact", split="test")

    assert loaded_names == ["beir/scifact/test"]
    assert documents == [
        {"doc_id": "d1", "text": "Title 1\nDocument one."},
        {"doc_id": "d2", "text": "Document two."},
        {"doc_id": "d3", "text": "Document three."},
    ]
    assert queries == [
        {"query_id": "q1", "text": "first query"},
        {"query_id": "q2", "text": "second query"},
        {"query_id": "q3", "text": "third query"},
    ]
    assert qrels == {
        "q1": {"d1": 2},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }


def test_load_ir_dataset_applies_document_and_query_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ir_datasets(monkeypatch)

    documents, queries, qrels = load_ir_dataset(
        "beir/scifact/test",
        limit_docs=2,
        limit_queries=1,
    )

    assert [document["doc_id"] for document in documents] == ["d1", "d2"]
    assert [query["query_id"] for query in queries] == ["q1"]
    assert qrels == {"q1": {"d1": 2}}


def test_load_ir_dataset_missing_dependency_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "ir_datasets", raising=False)
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ir_datasets":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Install it with `pip install ir_datasets`"):
        load_ir_dataset("beir/scifact/test")


def test_msmarco_without_limit_requires_explicit_large_download_flag() -> None:
    with pytest.raises(ValueError, match="Refusing to export full MS MARCO"):
        validate_ir_dataset_export_request(
            dataset_name="msmarco-passage/train",
            limit_docs=None,
            allow_large_download=False,
        )


def test_msmarco_allows_limit_or_explicit_large_download_flag() -> None:
    validate_ir_dataset_export_request(
        dataset_name="msmarco-passage/train",
        limit_docs=100,
        allow_large_download=False,
    )
    validate_ir_dataset_export_request(
        dataset_name="msmarco-passage/train",
        limit_docs=None,
        allow_large_download=True,
    )
