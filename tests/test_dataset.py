from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset import DatasetFormat, LocalJsonlDatasetLoader, load_dataset
from src.dataset import load_documents, load_qrels, load_queries


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_documents(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {"doc_id": "d1", "text": "First document."},
            {"doc_id": "d2", "text": "Second document."},
        ],
    )

    documents = load_documents(path)

    assert documents == [
        {"doc_id": "d1", "text": "First document."},
        {"doc_id": "d2", "text": "Second document."},
    ]


def test_load_queries(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"query_id": "q1", "text": "first query"},
            {"query_id": "q2", "text": "second query"},
        ],
    )

    queries = load_queries(path)

    assert queries == [
        {"query_id": "q1", "text": "first query"},
        {"query_id": "q2", "text": "second query"},
    ]


def test_load_qrels(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "qrels.jsonl",
        [
            {"query_id": "q1", "doc_id": "d1", "relevance": 2},
            {"query_id": "q1", "doc_id": "d2", "relevance": 0},
            {"query_id": "q2", "doc_id": "d3", "relevance": 1},
        ],
    )

    qrels = load_qrels(path)

    assert qrels == {
        "q1": {"d1": 2, "d2": 0},
        "q2": {"d3": 1},
    }


def test_local_jsonl_dataset_loader_from_paths(tmp_path: Path) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [{"doc_id": "d1", "text": "Document text."}],
    )
    queries_path = write_jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q1", "text": "query text"}],
    )
    qrels_path = write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_id": "q1", "doc_id": "d1", "relevance": 2}],
    )

    dataset = LocalJsonlDatasetLoader(
        documents_path=documents_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    ).load()

    assert dataset.documents == [{"doc_id": "d1", "text": "Document text."}]
    assert dataset.queries == [{"query_id": "q1", "text": "query text"}]
    assert dataset.qrels == {"q1": {"d1": 2}}


def test_load_dataset_local_jsonl_from_directory(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "documents.jsonl",
        [{"doc_id": "d1", "text": "Document text."}],
    )
    write_jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q1", "text": "query text"}],
    )
    write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_id": "q1", "doc_id": "d1", "relevance": 1}],
    )

    dataset = load_dataset(DatasetFormat.LOCAL_JSONL, dataset_dir=tmp_path)

    assert dataset.documents[0]["doc_id"] == "d1"
    assert dataset.queries[0]["query_id"] == "q1"
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_documents.jsonl"

    with pytest.raises(FileNotFoundError, match="JSONL file not found"):
        load_documents(missing_path)


def test_invalid_json_line_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(
        '{"doc_id": "d1", "text": "valid"}\n'
        '{"doc_id": "d2", "text": invalid}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Invalid JSON .* line 2"):
        load_documents(path)


def test_document_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "documents.jsonl",
        [{"doc_id": "d1"}],
    )

    with pytest.raises(ValueError, match="Missing 'text'"):
        load_documents(path)


def test_query_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q1"}],
    )

    with pytest.raises(ValueError, match="Missing 'text'"):
        load_queries(path)


def test_qrel_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_id": "q1", "doc_id": "d1"}],
    )

    with pytest.raises(ValueError, match="Missing 'relevance'"):
        load_qrels(path)
