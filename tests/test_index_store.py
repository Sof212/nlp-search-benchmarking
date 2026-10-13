from __future__ import annotations

from pathlib import Path

import pytest

from src.index_store import get_dataset_name, is_index_valid
from src.pipelines.build_indexes import load_or_build_retriever_index


def write_documents(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f'{{"doc_id": "{doc_id}", "text": "{text}"}}\n'
            for doc_id, text in rows
        ),
        encoding="utf-8",
    )
    return path


def test_building_bm25_index_creates_files(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    dataset_dir = tmp_path / "dataset"
    documents_path = write_documents(
        dataset_dir / "documents.jsonl",
        [("d1", "apple banana"), ("d2", "search ranking")],
    )
    output_dir = tmp_path / "indexes"

    result = load_or_build_retriever_index("bm25", dataset_dir, output_dir)

    assert result.status == "built"
    assert (result.index_dir / "manifest.json").exists()
    assert (result.index_dir / "index.pkl").exists()
    assert is_index_valid(result.index_dir, documents_path, "bm25")
    assert result.manifest["method"] == "bm25"
    assert result.manifest["dataset_name"] == get_dataset_name(dataset_dir)


def test_building_tfidf_index_creates_files(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    dataset_dir = tmp_path / "dataset"
    documents_path = write_documents(
        dataset_dir / "documents.jsonl",
        [("d1", "apple banana"), ("d2", "search ranking")],
    )
    output_dir = tmp_path / "indexes"

    result = load_or_build_retriever_index("tfidf", dataset_dir, output_dir)

    assert result.status == "built"
    assert (result.index_dir / "manifest.json").exists()
    assert (result.index_dir / "index.pkl").exists()
    assert is_index_valid(result.index_dir, documents_path, "tfidf")


def test_loading_existing_bm25_index_works(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    dataset_dir = tmp_path / "dataset"
    write_documents(
        dataset_dir / "documents.jsonl",
        [("d1", "apple banana"), ("d2", "search ranking")],
    )
    output_dir = tmp_path / "indexes"

    built = load_or_build_retriever_index("bm25", dataset_dir, output_dir)
    loaded = load_or_build_retriever_index("bm25", dataset_dir, output_dir)

    assert built.index_dir == loaded.index_dir
    assert loaded.status == "loaded"
    results = loaded.retriever.search("apple", top_k=1)
    assert results[0]["doc_id"] == "d1"


def test_stale_index_is_invalid_when_documents_change(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    dataset_dir = tmp_path / "dataset"
    documents_path = write_documents(
        dataset_dir / "documents.jsonl",
        [("d1", "apple banana"), ("d2", "search ranking")],
    )
    output_dir = tmp_path / "indexes"

    result = load_or_build_retriever_index("bm25", dataset_dir, output_dir)
    documents_path.write_text(
        '{"doc_id": "d1", "text": "changed document"}\n',
        encoding="utf-8",
    )

    assert not is_index_valid(result.index_dir, documents_path, "bm25")


def test_build_pipeline_missing_documents_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset documents file not found"):
        load_or_build_retriever_index(
            "bm25",
            tmp_path / "missing_dataset",
            tmp_path / "indexes",
        )
