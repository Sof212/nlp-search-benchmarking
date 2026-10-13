"""Evaluate a retrieval run against qrels and save metrics as JSON."""

from __future__ import annotations

import argparse
from typing import Any

from src.dataset import load_qrels
from src.evaluation import evaluate_run
from src.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", required=True, help="Path to qrels JSONL.")
    parser.add_argument("--run", required=True, help="Path to run JSON.")
    parser.add_argument("--output", required=True, help="Path to metrics JSON.")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10, 100],
        help="Evaluation cutoffs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qrels = load_qrels(args.qrels)
    run: Any = load_json(args.run)
    if not isinstance(run, dict):
        raise ValueError(f"Expected run JSON object in {args.run}")
    metrics = evaluate_run(qrels, run, k_values=args.k_values)
    save_json(metrics, args.output)


if __name__ == "__main__":
    main()
