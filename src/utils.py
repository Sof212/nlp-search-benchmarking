"""Shared filesystem, configuration, and ranking helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory for ``path`` exists."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON file."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write data as deterministic UTF-8 JSON."""
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict when no path is provided."""
    if path is None:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install pyyaml to read YAML config files") from exc
    with yaml_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in config file: {yaml_path}")
    return loaded


def normalize_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    """Min-max normalize result scores by document ID."""
    if not results:
        return {}
    scores = [float(result.get("score", 0.0)) for result in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {str(result["doc_id"]): 1.0 for result in results}
    return {
        str(result["doc_id"]): (float(result.get("score", 0.0)) - min_score)
        / (max_score - min_score)
        for result in results
    }


def sorted_results(
    results: Iterable[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Sort retrieval results by descending score and stable document ID tie-break."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return sorted(
        results,
        key=lambda result: (-float(result.get("score", 0.0)), str(result["doc_id"])),
    )[:top_k]
