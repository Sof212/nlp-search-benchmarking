"""Validate local datasets or export ir_datasets datasets to local JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.dataset import (
    load_documents,
    load_ir_dataset,
    load_qrels,
    load_queries,
    save_documents_jsonl,
    save_qrels_jsonl,
    save_queries_jsonl,
    validate_ir_dataset_export_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        choices=["local", "ir_datasets"],
        help="Dataset source type.",
    )
    parser.add_argument("--documents", help="Local documents JSONL path.")
    parser.add_argument("--queries", help="Local queries JSONL path.")
    parser.add_argument("--qrels", help="Local qrels JSONL path.")
    parser.add_argument(
        "--dataset-name",
        help="Dataset name passed directly to ir_datasets.load().",
    )
    parser.add_argument("--split", help="Optional split suffix for ir_datasets.")
    parser.add_argument("--output-dir", help="Directory for exported JSONL files.")
    parser.add_argument("--limit-docs", type=int, help="Maximum documents to export.")
    parser.add_argument("--limit-queries", type=int, help="Maximum queries to export.")
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Allow full MS MARCO export without --limit-docs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "local":
        _validate_local(args)
        return
    _export_ir_dataset(args)


def _validate_local(args: argparse.Namespace) -> None:
    if not args.documents or not args.queries or not args.qrels:
        raise ValueError(
            "--documents, --queries, and --qrels are required with --source local"
        )
    documents = load_documents(args.documents)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    print(
        "Validated local dataset: "
        f"{len(documents)} documents, {len(queries)} queries, {len(qrels)} qrel queries"
    )


def _export_ir_dataset(args: argparse.Namespace) -> None:
    if not args.dataset_name:
        raise ValueError("--dataset-name is required with --source ir_datasets")
    if not args.output_dir:
        raise ValueError("--output-dir is required with --source ir_datasets")
    if _is_msmarco_name(args.dataset_name) and args.limit_docs is None:
        print(
            "Warning: MS MARCO is large. Use --limit-docs for test exports or "
            "--allow-large-download for a full export.",
            file=sys.stderr,
        )
    validate_ir_dataset_export_request(
        dataset_name=args.dataset_name,
        limit_docs=args.limit_docs,
        allow_large_download=args.allow_large_download,
    )

    documents, queries, qrels = load_ir_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
    )
    output_dir = Path(args.output_dir)
    save_documents_jsonl(documents, output_dir / "documents.jsonl")
    save_queries_jsonl(queries, output_dir / "queries.jsonl")
    save_qrels_jsonl(qrels, output_dir / "qrels.jsonl")
    print(
        "Exported ir_datasets dataset: "
        f"{len(documents)} documents, {len(queries)} queries, {len(qrels)} qrel queries"
    )


def _is_msmarco_name(dataset_name: str) -> bool:
    normalized = dataset_name.lower().replace("-", "").replace("_", "")
    return "msmarco" in normalized


if __name__ == "__main__":
    main()
