"""Build persistent retriever indexes for prepared local datasets."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset import load_documents
from src.index_store import (
    build_manifest,
    config_key_from_config,
    get_dataset_name,
    get_index_dir,
    is_index_valid,
    load_manifest,
    load_retriever_index,
    save_retriever_index,
    write_manifest,
)
from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense_embedding import DenseEmbeddingRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.splade import SpladeRetriever
from src.retrievers.tfidf import TfidfRetriever
from src.utils import load_yaml_config


@dataclass(frozen=True)
class IndexedRetrieverResult:
    """Result returned when loading or building a persistent index."""

    retriever: BaseRetriever
    index_dir: Path
    status: str
    manifest: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Prepared dataset directory containing documents.jsonl.",
    )
    parser.add_argument(
        "--methods",
        required=True,
        nargs="+",
        choices=["bm25", "tfidf", "dense", "splade", "hybrid"],
        help="Retriever indexes to build.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/indexes",
        help="Root directory where persistent indexes are stored.",
    )
    parser.add_argument("--config", help="Optional retriever YAML config.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for transformer-backed methods.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a valid saved index already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    for method in args.methods:
        _validate_config_method(method, config, args.config)
        print(f"[build-indexes] Preparing {method} index...")
        result = load_or_build_retriever_index(
            method=method,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            config=config,
            config_path=args.config,
            device=_resolve_cli_device(args.device),
            force=args.force,
        )
        print(
            f"[build-indexes] {method}: {result.status} index at "
            f"{result.index_dir}"
        )


def load_or_build_retriever_index(
    method: str,
    dataset_dir: str | Path,
    output_dir: str | Path = "outputs/indexes",
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    device: str | None = None,
    force: bool = False,
) -> IndexedRetrieverResult:
    """Load a valid persistent index, or build and save it."""
    dataset_path = Path(dataset_dir)
    documents_path = dataset_path / "documents.jsonl"
    if not documents_path.exists():
        raise FileNotFoundError(
            f"Dataset documents file not found: {documents_path}"
        )

    retriever_config = effective_retriever_config(method, config, device=device)
    persistent_config = persistent_config_for_key(method, retriever_config)
    config_key = config_key_from_config(persistent_config)
    dataset_name = get_dataset_name(dataset_path)
    index_dir = get_index_dir(Path(output_dir), dataset_name, method, config_key)

    if not force and is_index_valid(index_dir, documents_path, method, config_key):
        retriever = load_retriever_index(index_dir, method)
        _apply_runtime_device(retriever, method, device)
        return IndexedRetrieverResult(
            retriever=retriever,
            index_dir=index_dir,
            status="loaded",
            manifest=load_manifest(index_dir),
        )

    documents = load_documents(documents_path)
    retriever, already_indexed = _build_retriever(
        method=method,
        retriever_config=retriever_config,
        dataset_dir=dataset_path,
        output_dir=Path(output_dir),
        config_path=config_path,
        device=device,
        force=force,
    )
    if not already_indexed:
        retriever.index(documents)

    manifest = build_manifest(
        dataset_name=dataset_name,
        documents_path=documents_path,
        method=method,
        num_documents=len(documents),
        config_path=config_path,
        config_key=config_key,
        model_name=_model_name_for_manifest(method, retriever_config),
    )
    save_retriever_index(retriever, method, index_dir)
    write_manifest(index_dir, manifest)
    return IndexedRetrieverResult(
        retriever=retriever,
        index_dir=index_dir,
        status="built",
        manifest=manifest,
    )


def get_index_status(
    method: str,
    dataset_dir: str | Path,
    output_dir: str | Path = "outputs/indexes",
    config: dict[str, Any] | None = None,
    device: str | None = None,
) -> tuple[bool, Path]:
    """Return whether a valid index exists and where it is stored."""
    dataset_path = Path(dataset_dir)
    documents_path = dataset_path / "documents.jsonl"
    retriever_config = effective_retriever_config(method, config, device=device)
    config_key = config_key_from_config(
        persistent_config_for_key(method, retriever_config)
    )
    index_dir = get_index_dir(
        Path(output_dir),
        get_dataset_name(dataset_path),
        method,
        config_key,
    )
    return is_index_valid(index_dir, documents_path, method, config_key), index_dir


def effective_retriever_config(
    method: str,
    config: dict[str, Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Return explicit retriever config used for construction and keying."""
    raw_config = deepcopy((config or {}).get("retriever", {}))
    if method == "dense":
        effective = {
            "model_name": "intfloat/e5-small-v2",
            "batch_size": 32,
            "normalize_embeddings": True,
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
            "device": None,
        }
        effective.update(raw_config)
        effective["device"] = device
        return effective
    if method == "splade":
        effective = {
            "model_name": "naver/splade-cocondenser-ensembledistil",
            "batch_size": 8,
            "max_length": 256,
            "device": None,
        }
        effective.update(raw_config)
        effective["device"] = device
        return effective
    if method == "hybrid":
        effective = {
            "method": "rrf",
            "candidate_pool_size": 100,
            "rrf_k": 60,
            "alpha": 0.5,
            "beta": 0.5,
            "bm25_config": {},
            "dense_config": {
                "model_name": "intfloat/e5-small-v2",
                "batch_size": 32,
                "normalize_embeddings": True,
                "query_prefix": "query: ",
                "document_prefix": "passage: ",
                "device": None,
            },
        }
        dense_config = dict(effective["dense_config"])
        bm25_config = dict(effective["bm25_config"])
        if "dense_config" in raw_config:
            dense_config.update(raw_config.pop("dense_config") or {})
        if "bm25_config" in raw_config:
            bm25_config.update(raw_config.pop("bm25_config") or {})
        effective.update(raw_config)
        dense_config["device"] = device
        effective["dense_config"] = dense_config
        effective["bm25_config"] = bm25_config
        return effective
    return raw_config


def persistent_config_for_key(method: str, retriever_config: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only fields and default lexical configs from index keys."""
    config = deepcopy(retriever_config)
    if method in {"dense", "splade"}:
        config.pop("device", None)
        return config
    if method == "hybrid":
        dense_config = dict(config.get("dense_config") or {})
        dense_config.pop("device", None)
        config["dense_config"] = dense_config
        return config
    return config


def _build_retriever(
    method: str,
    retriever_config: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    config_path: str | None,
    device: str | None,
    force: bool,
) -> tuple[BaseRetriever, bool]:
    if method == "tfidf":
        return TfidfRetriever(**retriever_config), False
    if method == "bm25":
        return BM25Retriever(**retriever_config), False
    if method == "dense":
        return DenseEmbeddingRetriever(**retriever_config), False
    if method == "splade":
        return SpladeRetriever(**retriever_config), False
    if method == "hybrid":
        bm25_config = _component_bm25_config(retriever_config)
        dense_config = _component_dense_config(retriever_config, device)
        bm25_result = load_or_build_retriever_index(
            method="bm25",
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            config={"retriever": bm25_config} if bm25_config else None,
            config_path=config_path if bm25_config else None,
            device=None,
            force=force,
        )
        dense_result = load_or_build_retriever_index(
            method="dense",
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            config={"retriever": dense_config},
            config_path=config_path,
            device=device,
            force=force,
        )
        hybrid_kwargs = {
            key: value
            for key, value in retriever_config.items()
            if key not in {"bm25_config", "dense_config"}
        }
        return (
            HybridRetriever(
                sparse_retriever=bm25_result.retriever,
                dense_retriever=dense_result.retriever,
                **hybrid_kwargs,
            ),
            True,
        )
    raise ValueError(f"Unsupported retrieval method: {method}")


def _component_bm25_config(retriever_config: dict[str, Any]) -> dict[str, Any]:
    bm25_config = dict(retriever_config.get("bm25_config") or {})
    default_bm25 = {"k1": 1.5, "b": 0.75}
    if bm25_config == default_bm25:
        return {}
    return bm25_config


def _component_dense_config(
    retriever_config: dict[str, Any],
    device: str | None,
) -> dict[str, Any]:
    dense_config = dict(retriever_config.get("dense_config") or {})
    dense_config["device"] = device
    return dense_config


def _apply_runtime_device(retriever: BaseRetriever, method: str, device: str | None) -> None:
    if method in {"dense", "splade"} and hasattr(retriever, "device"):
        setattr(retriever, "device", device)
    if method == "hybrid":
        dense_retriever = getattr(retriever, "dense_retriever", None)
        if dense_retriever is not None and hasattr(dense_retriever, "device"):
            setattr(dense_retriever, "device", device)


def _model_name_for_manifest(method: str, retriever_config: dict[str, Any]) -> str | None:
    if method in {"dense", "splade"}:
        return str(retriever_config.get("model_name"))
    if method == "hybrid":
        dense_config = retriever_config.get("dense_config") or {}
        return f"BM25 + {dense_config.get('model_name', 'dense')}"
    return None


def _validate_config_method(
    method: str,
    config: dict[str, Any],
    config_path: str | None,
) -> None:
    if not config_path:
        return
    configured_method = config.get("method")
    if configured_method and configured_method != method:
        raise ValueError(
            f"Config method '{configured_method}' does not match requested "
            f"method '{method}'"
        )


def _resolve_cli_device(device: str) -> str | None:
    return None if device == "auto" else device


if __name__ == "__main__":
    main()
