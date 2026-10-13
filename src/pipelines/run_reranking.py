"""Run cross-encoder reranking over a first-stage retrieval run."""

from __future__ import annotations

import argparse
from typing import Any

from src.dataset import documents_by_id, load_documents, load_queries
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.utils import load_json, load_yaml_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, help="Path to retrieval run JSON.")
    parser.add_argument("--documents", required=True, help="Path to documents JSONL.")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL.")
    parser.add_argument("--output", required=True, help="Path to reranked run JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Reranked results per query.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="Number of first-stage candidates to rerank per query.",
    )
    parser.add_argument("--config", help="Optional YAML config file.")
    parser.add_argument("--model-name", help="Override cross-encoder model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be greater than or equal to --top-k")

    config = load_yaml_config(args.config)
    reranker_config: dict[str, Any] = dict(config.get("reranker", config))
    if args.model_name:
        reranker_config["model_name"] = args.model_name

    input_run = load_json(args.input_run)
    if not isinstance(input_run, dict):
        raise ValueError(f"Expected run JSON object in {args.input_run}")

    documents = documents_by_id(load_documents(args.documents))
    queries = load_queries(args.queries)
    reranker = CrossEncoderReranker(**reranker_config)

    reranked_run: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        candidates = _hydrate_candidates(
            input_run.get(query_id, [])[: args.candidate_k],
            documents,
        )
        reranked_run[query_id] = reranker.rerank(
            str(query["text"]),
            candidates,
            top_k=args.top_k,
        )
    save_json(reranked_run, args.output)


def _hydrate_candidates(
    candidates: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for candidate in candidates:
        doc_id = str(candidate.get("doc_id", ""))
        if not doc_id:
            raise ValueError("Run candidate is missing doc_id")
        if "text" in candidate:
            hydrated.append(dict(candidate))
            continue
        document = documents.get(doc_id)
        if document is None:
            raise ValueError(f"Run references unknown doc_id: {doc_id}")
        item = dict(candidate)
        item["text"] = str(document["text"])
        hydrated.append(item)
    return hydrated


if __name__ == "__main__":
    main()
