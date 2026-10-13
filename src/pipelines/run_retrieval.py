"""Run first-stage retrieval and save a benchmark run JSON file."""

from __future__ import annotations

import argparse
from typing import Any

from src.dataset import load_documents, load_queries
from src.retrievers.adaptive_splade_hybrid import AdaptiveSpladeHybridRetriever
from src.retrievers.base import BaseRetriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense_embedding import DenseEmbeddingRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.splade import SpladeRetriever
from src.retrievers.tfidf import TfidfRetriever
from src.utils import load_yaml_config, save_json


def build_retriever(method: str, config: dict[str, Any]) -> BaseRetriever:
    """Build a retriever from a method name and config mapping."""
    if method == "adaptive_splade_hybrid_rerank":
        return AdaptiveSpladeHybridRetriever.from_config(config)
    retriever_config = dict(config.get("retriever", {}))
    if method == "tfidf":
        return TfidfRetriever(**retriever_config)
    if method == "bm25":
        return BM25Retriever(**retriever_config)
    if method == "dense":
        return DenseEmbeddingRetriever(**retriever_config)
    if method == "splade":
        return SpladeRetriever(**retriever_config)
    if method == "hybrid":
        return HybridRetriever(**retriever_config)
    raise ValueError(f"Unsupported retrieval method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "tfidf",
            "bm25",
            "splade",
            "dense",
            "hybrid",
            "adaptive_splade_hybrid_rerank",
        ],
        help="Retrieval method to run.",
    )
    parser.add_argument("--documents", required=True, help="Path to documents JSONL.")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL.")
    parser.add_argument("--output", required=True, help="Path to output run JSON.")
    parser.add_argument("--top-k", type=int, default=100, help="Results per query.")
    parser.add_argument("--config", help="Optional YAML config file.")
    parser.add_argument(
        "--model-name",
        help="Override model_name for dense, splade, or hybrid dense component.",
    )
    parser.add_argument(
        "--fusion-method",
        choices=["rrf", "weighted"],
        help="Override hybrid fusion method.",
    )
    parser.add_argument("--alpha", type=float, help="Override hybrid sparse weight.")
    parser.add_argument("--beta", type=float, help="Override hybrid dense weight.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    method = str(config.get("method", args.method))
    if method != args.method:
        raise ValueError(
            f"Config method '{method}' does not match CLI method '{args.method}'"
        )
    config.setdefault("retriever", {})
    _apply_cli_overrides(config["retriever"], args)

    documents = load_documents(args.documents)
    queries = load_queries(args.queries)
    retriever = build_retriever(args.method, config)
    retriever.index(documents)
    run = retriever.batch_search(queries, top_k=args.top_k)
    save_json(run, args.output)


def _apply_cli_overrides(
    retriever_config: dict[str, Any], args: argparse.Namespace
) -> None:
    if args.model_name:
        if args.method == "hybrid":
            retriever_config.setdefault("dense_config", {})["model_name"] = args.model_name
        else:
            retriever_config["model_name"] = args.model_name
    if args.fusion_method:
        retriever_config["method"] = args.fusion_method
    if args.alpha is not None:
        retriever_config["alpha"] = args.alpha
    if args.beta is not None:
        retriever_config["beta"] = args.beta


if __name__ == "__main__":
    main()
