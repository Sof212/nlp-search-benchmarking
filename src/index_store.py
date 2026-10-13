"""Persistent retriever index storage helpers."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ARRAY_INDEX_FILES = {
    "dense": ("_document_embeddings", "embeddings.npy"),
    "splade": ("_document_vectors", "vectors.npy"),
}


def compute_file_fingerprint(path: Path) -> str:
    """Return a SHA-256 fingerprint for a file's byte content."""
    if not path.exists():
        raise FileNotFoundError(f"Cannot fingerprint missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_dataset_name(dataset_dir: Path) -> str:
    """Return a stable dataset name from a dataset directory."""
    return dataset_dir.resolve().name


def get_index_dir(
    output_dir: Path,
    dataset_name: str,
    method: str,
    config_key: str | None = None,
) -> Path:
    """Return the index directory for a dataset/method/config combination."""
    base = output_dir / _safe_path_part(dataset_name) / _safe_path_part(method)
    if config_key:
        return base / _safe_path_part(config_key)
    return base


def index_exists(index_dir: Path) -> bool:
    """Return whether an index directory contains a manifest and index pickle."""
    return (index_dir / "manifest.json").exists() and (index_dir / "index.pkl").exists()


def load_manifest(index_dir: Path) -> dict[str, Any]:
    """Load an index manifest."""
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected JSON object in manifest: {manifest_path}")
    return manifest


def is_index_valid(
    index_dir: Path,
    documents_path: Path,
    method: str,
    config_key: str | None = None,
) -> bool:
    """Return whether an existing index matches the current documents and config."""
    if not index_exists(index_dir):
        return False
    try:
        manifest = load_manifest(index_dir)
        current_fingerprint = compute_file_fingerprint(documents_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("method") == method
        and manifest.get("config_key") == config_key
        and manifest.get("documents_fingerprint") == current_fingerprint
    )


def save_pickle(obj: Any, path: Path) -> None:
    """Save an object with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Load a pickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def write_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    """Write an index manifest as deterministic JSON."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def build_manifest(
    dataset_name: str,
    documents_path: Path,
    method: str,
    num_documents: int,
    config_path: str | None = None,
    config_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build a standard index manifest."""
    return {
        "dataset_name": dataset_name,
        "documents_path": str(documents_path),
        "method": method,
        "config_path": config_path,
        "config_key": config_key,
        "model_name": model_name,
        "documents_fingerprint": compute_file_fingerprint(documents_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_documents": num_documents,
    }


def save_retriever_index(retriever: Any, method: str, index_dir: Path) -> None:
    """Save a retriever index.

    Dense and SPLADE arrays are stored as ``.npy`` files so model objects are not
    pickled. The retriever object is saved without the heavyweight loaded model.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    array_attr, array_filename = ARRAY_INDEX_FILES.get(method, (None, None))
    array_value = getattr(retriever, array_attr, None) if array_attr else None
    cleared_objects = _clear_loaded_model_objects(retriever)
    try:
        if array_attr and array_value is not None and array_filename:
            np.save(index_dir / array_filename, array_value)
            setattr(retriever, array_attr, None)
        save_pickle(retriever, index_dir / "index.pkl")
    finally:
        if array_attr:
            setattr(retriever, array_attr, array_value)
        _restore_loaded_model_objects(cleared_objects)


def load_retriever_index(index_dir: Path, method: str) -> Any:
    """Load a persisted retriever index."""
    retriever = load_pickle(index_dir / "index.pkl")
    array_attr, array_filename = ARRAY_INDEX_FILES.get(method, (None, None))
    if array_attr and array_filename:
        array_path = index_dir / array_filename
        if not array_path.exists():
            raise FileNotFoundError(f"Index array file not found: {array_path}")
        setattr(retriever, array_attr, np.load(array_path))
    return retriever


def config_key_from_config(config: dict[str, Any] | None) -> str | None:
    """Return a short stable hash for a retriever config."""
    if not config:
        return None
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("._")
    return cleaned or "default"


def _clear_loaded_model_objects(retriever: Any) -> list[tuple[Any, str, Any]]:
    cleared: list[tuple[Any, str, Any]] = []
    objects = [
        retriever,
        getattr(retriever, "sparse_retriever", None),
        getattr(retriever, "dense_retriever", None),
    ]
    for obj in objects:
        if obj is None:
            continue
        for attr in ("_model", "_tokenizer"):
            if hasattr(obj, attr):
                cleared.append((obj, attr, getattr(obj, attr)))
                setattr(obj, attr, None)
    return cleared


def _restore_loaded_model_objects(cleared: list[tuple[Any, str, Any]]) -> None:
    for obj, attr, value in cleared:
        setattr(obj, attr, value)
