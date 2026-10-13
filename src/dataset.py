"""Dataset loading utilities and loader interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

JsonRecord = dict[str, Any]
Qrels = dict[str, dict[str, int]]


class DatasetFormat(str, Enum):
    """Supported dataset format identifiers."""

    LOCAL_JSONL = "local_jsonl"
    IR_DATASETS = "ir_datasets"
    BEIR = "beir"
    MS_MARCO = "ms_marco"


@dataclass(frozen=True)
class SearchDataset:
    """In-memory benchmark dataset used by retrieval and evaluation pipelines."""

    documents: list[JsonRecord]
    queries: list[JsonRecord]
    qrels: Qrels


class DatasetLoader(ABC):
    """Base interface for benchmark dataset loaders."""

    @abstractmethod
    def load(self) -> SearchDataset:
        """Load a dataset into the common benchmark representation."""
        pass


class LocalJsonlDatasetLoader(DatasetLoader):
    """Load the local simple JSONL dataset format.

    The expected files are:
    - ``documents.jsonl`` with ``doc_id`` and ``text`` fields
    - ``queries.jsonl`` with ``query_id`` and ``text`` fields
    - ``qrels.jsonl`` with ``query_id``, ``doc_id``, and ``relevance`` fields
    """

    def __init__(
        self,
        documents_path: str | Path,
        queries_path: str | Path,
        qrels_path: str | Path,
    ) -> None:
        self.documents_path = Path(documents_path)
        self.queries_path = Path(queries_path)
        self.qrels_path = Path(qrels_path)

    @classmethod
    def from_directory(
        cls,
        dataset_dir: str | Path,
        documents_filename: str = "documents.jsonl",
        queries_filename: str = "queries.jsonl",
        qrels_filename: str = "qrels.jsonl",
    ) -> "LocalJsonlDatasetLoader":
        """Build a local loader from a directory containing standard filenames."""
        root = Path(dataset_dir)
        return cls(
            documents_path=root / documents_filename,
            queries_path=root / queries_filename,
            qrels_path=root / qrels_filename,
        )

    def load(self) -> SearchDataset:
        """Load local documents, queries, and qrels."""
        return SearchDataset(
            documents=load_documents(self.documents_path),
            queries=load_queries(self.queries_path),
            qrels=load_qrels(self.qrels_path),
        )


class IrDatasetsLoader(DatasetLoader):
    """Load BEIR, MS MARCO, or another dataset exposed by ``ir_datasets``."""

    def __init__(
        self,
        dataset_name: str,
        split: str | None = None,
        limit_docs: int | None = None,
        limit_queries: int | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.limit_docs = limit_docs
        self.limit_queries = limit_queries

    def load(self) -> SearchDataset:
        """Load an ``ir_datasets`` dataset into the benchmark representation."""
        documents, queries, qrels = load_ir_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            limit_docs=self.limit_docs,
            limit_queries=self.limit_queries,
        )
        return SearchDataset(documents=documents, queries=queries, qrels=qrels)


def build_dataset_loader(
    dataset_format: str | DatasetFormat,
    **kwargs: Any,
) -> DatasetLoader:
    """Build a dataset loader for a supported dataset format.

    Local JSONL stays the default. BEIR and MS MARCO are loaded through the
    generic ``ir_datasets`` path by passing their dataset names directly.
    """
    normalized_format = _normalize_dataset_format(dataset_format)
    if normalized_format is DatasetFormat.LOCAL_JSONL:
        dataset_dir = kwargs.get("dataset_dir")
        if dataset_dir is not None:
            return LocalJsonlDatasetLoader.from_directory(
                dataset_dir=dataset_dir,
                documents_filename=kwargs.get(
                    "documents_filename", "documents.jsonl"
                ),
                queries_filename=kwargs.get("queries_filename", "queries.jsonl"),
                qrels_filename=kwargs.get("qrels_filename", "qrels.jsonl"),
            )
        return LocalJsonlDatasetLoader(
            documents_path=_require_loader_arg(kwargs, "documents_path"),
            queries_path=_require_loader_arg(kwargs, "queries_path"),
            qrels_path=_require_loader_arg(kwargs, "qrels_path"),
        )
    if normalized_format in {
        DatasetFormat.IR_DATASETS,
        DatasetFormat.BEIR,
        DatasetFormat.MS_MARCO,
    }:
        return IrDatasetsLoader(
            dataset_name=_require_string_arg(kwargs, "dataset_name"),
            split=kwargs.get("split"),
            limit_docs=kwargs.get("limit_docs"),
            limit_queries=kwargs.get("limit_queries"),
        )
    raise ValueError(f"Unsupported dataset format: {dataset_format!r}")


def load_dataset(
    dataset_format: str | DatasetFormat = DatasetFormat.LOCAL_JSONL,
    **kwargs: Any,
) -> SearchDataset:
    """Load a complete benchmark dataset with the selected loader."""
    return build_dataset_loader(dataset_format, **kwargs).load()


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    """Read a JSONL file into a list of dictionaries.

    Args:
        path: Path to a UTF-8 encoded JSONL file.

    Returns:
        Parsed JSON records in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is invalid JSON or is not an object.
    """
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    records: list[JsonRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object in {jsonl_path} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def write_jsonl(records: list[JsonRecord], path: str | Path) -> None:
    """Write JSON records to a UTF-8 JSONL file."""
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def load_documents(path: str | Path) -> list[JsonRecord]:
    """Load benchmark documents from JSONL.

    Expected schema per line:
        {"doc_id": "doc_1", "text": "Document content here."}
    """
    documents = read_jsonl(path)
    seen_doc_ids: set[str] = set()
    for index, document in enumerate(documents, start=1):
        _require_string(document, "doc_id", path, index)
        _require_string(document, "text", path, index)
        doc_id = document["doc_id"]
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id in {path}: {doc_id}")
        seen_doc_ids.add(doc_id)
    return documents


def save_documents_jsonl(documents: list[JsonRecord], path: str | Path) -> None:
    """Save document records in the local benchmark JSONL format."""
    for index, document in enumerate(documents, start=1):
        _require_string(document, "doc_id", path, index)
        _require_string(document, "text", path, index)
    write_jsonl(documents, path)


def load_queries(path: str | Path) -> list[JsonRecord]:
    """Load benchmark queries from JSONL.

    Expected schema per line:
        {"query_id": "q_1", "text": "User query here."}
    """
    queries = read_jsonl(path)
    seen_query_ids: set[str] = set()
    for index, query in enumerate(queries, start=1):
        _require_string(query, "query_id", path, index)
        _require_string(query, "text", path, index)
        query_id = query["query_id"]
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id in {path}: {query_id}")
        seen_query_ids.add(query_id)
    return queries


def save_queries_jsonl(queries: list[JsonRecord], path: str | Path) -> None:
    """Save query records in the local benchmark JSONL format."""
    for index, query in enumerate(queries, start=1):
        _require_string(query, "query_id", path, index)
        _require_string(query, "text", path, index)
    write_jsonl(queries, path)


def load_qrels(path: str | Path) -> Qrels:
    """Load relevance judgments from JSONL into a nested dictionary.

    Expected schema per line:
        {"query_id": "q_1", "doc_id": "doc_1", "relevance": 1}
    """
    records = read_jsonl(path)
    qrels: Qrels = {}
    for index, record in enumerate(records, start=1):
        _require_string(record, "query_id", path, index)
        _require_string(record, "doc_id", path, index)
        if "relevance" not in record:
            raise ValueError(f"Missing 'relevance' in {path} at record {index}")
        relevance = record["relevance"]
        if not isinstance(relevance, int) or relevance < 0:
            raise ValueError(
                f"Expected non-negative integer relevance in {path} at record {index}"
            )
        qrels.setdefault(record["query_id"], {})[record["doc_id"]] = relevance
    return qrels


def save_qrels_jsonl(qrels: Qrels, path: str | Path) -> None:
    """Save qrels in JSONL form with one query/document judgment per line."""
    records: list[JsonRecord] = []
    for query_id in sorted(qrels):
        for doc_id in sorted(qrels[query_id]):
            relevance = qrels[query_id][doc_id]
            if not isinstance(relevance, int) or relevance < 0:
                raise ValueError(
                    f"Expected non-negative integer relevance for "
                    f"query_id={query_id!r}, doc_id={doc_id!r}"
                )
            records.append(
                {
                    "query_id": str(query_id),
                    "doc_id": str(doc_id),
                    "relevance": relevance,
                }
            )
    write_jsonl(records, path)


def load_ir_dataset(
    dataset_name: str,
    split: str | None = None,
    limit_docs: int | None = None,
    limit_queries: int | None = None,
) -> tuple[list[JsonRecord], list[JsonRecord], Qrels]:
    """Load an ``ir_datasets`` dataset into the local benchmark representation.

    Args:
        dataset_name: Any dataset name accepted by ``ir_datasets.load``.
        split: Optional suffix to append to ``dataset_name``.
        limit_docs: Optional maximum number of documents to read.
        limit_queries: Optional maximum number of queries to read.

    Returns:
        ``(documents, queries, qrels)`` in the project's local JSONL-compatible
        format.
    """
    _validate_optional_limit(limit_docs, "limit_docs")
    _validate_optional_limit(limit_queries, "limit_queries")
    resolved_name = resolve_ir_dataset_name(dataset_name, split)
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError(
            "ir_datasets is required to load benchmark datasets. "
            "Install it with `pip install ir_datasets`."
        ) from exc

    dataset = ir_datasets.load(resolved_name)
    documents = _convert_ir_documents(dataset.docs_iter(), limit_docs)
    queries = _convert_ir_queries(dataset.queries_iter(), limit_queries)
    qrels = _convert_ir_qrels(
        dataset.qrels_iter(),
        allowed_doc_ids={document["doc_id"] for document in documents}
        if limit_docs is not None
        else None,
        allowed_query_ids={query["query_id"] for query in queries}
        if limit_queries is not None
        else None,
    )
    return documents, queries, qrels


def resolve_ir_dataset_name(dataset_name: str, split: str | None = None) -> str:
    """Resolve an ``ir_datasets`` dataset name and optional split suffix."""
    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")
    normalized = dataset_name.strip("/")
    if not split:
        return normalized
    split_suffix = split.strip("/")
    if not split_suffix:
        return normalized
    if normalized.endswith(f"/{split_suffix}") or normalized == split_suffix:
        return normalized
    return f"{normalized}/{split_suffix}"


def validate_ir_dataset_export_request(
    dataset_name: str,
    limit_docs: int | None,
    allow_large_download: bool = False,
) -> None:
    """Guard against accidental full MS MARCO exports."""
    _validate_optional_limit(limit_docs, "limit_docs")
    normalized_name = dataset_name.lower().replace("-", "").replace("_", "")
    if "msmarco" in normalized_name and limit_docs is None and not allow_large_download:
        raise ValueError(
            "Refusing to export full MS MARCO without --limit-docs. "
            "Pass --allow-large-download to override intentionally."
        )


def documents_by_id(documents: list[JsonRecord]) -> dict[str, JsonRecord]:
    """Return documents keyed by ``doc_id`` with duplicate protection."""
    indexed: dict[str, JsonRecord] = {}
    for document in documents:
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str):
            raise ValueError("Every document must contain a string doc_id")
        if doc_id in indexed:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        indexed[doc_id] = document
    return indexed


def queries_by_id(queries: list[JsonRecord]) -> dict[str, JsonRecord]:
    """Return queries keyed by ``query_id`` with duplicate protection."""
    indexed: dict[str, JsonRecord] = {}
    for query in queries:
        query_id = query.get("query_id")
        if not isinstance(query_id, str):
            raise ValueError("Every query must contain a string query_id")
        if query_id in indexed:
            raise ValueError(f"Duplicate query_id: {query_id}")
        indexed[query_id] = query
    return indexed


def _require_string(
    record: JsonRecord, field_name: str, path: str | Path, record_index: int
) -> None:
    if field_name not in record:
        raise ValueError(f"Missing '{field_name}' in {path} at record {record_index}")
    if not isinstance(record[field_name], str) or not record[field_name]:
        raise ValueError(
            f"Expected non-empty string '{field_name}' in {path} at record "
            f"{record_index}"
        )


def _normalize_dataset_format(dataset_format: str | DatasetFormat) -> DatasetFormat:
    if isinstance(dataset_format, DatasetFormat):
        return dataset_format
    aliases = {
        "local": DatasetFormat.LOCAL_JSONL,
        "jsonl": DatasetFormat.LOCAL_JSONL,
        "local_jsonl": DatasetFormat.LOCAL_JSONL,
        "ir_datasets": DatasetFormat.IR_DATASETS,
        "ir-datasets": DatasetFormat.IR_DATASETS,
        "beir": DatasetFormat.BEIR,
        "ms_marco": DatasetFormat.MS_MARCO,
        "msmarco": DatasetFormat.MS_MARCO,
    }
    try:
        return aliases[dataset_format.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset format: {dataset_format!r}") from exc


def _require_loader_arg(kwargs: dict[str, Any], name: str) -> str | Path:
    value = kwargs.get(name)
    if value is None:
        raise ValueError(f"Missing required loader argument: {name}")
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Loader argument '{name}' must be a path-like value")
    return value


def _require_string_arg(kwargs: dict[str, Any], name: str) -> str:
    value = kwargs.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string loader argument: {name}")
    return value


def _validate_optional_limit(limit: int | None, name: str) -> None:
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError(f"{name} must be a positive integer when provided")


def _convert_ir_documents(documents_iter: Any, limit: int | None) -> list[JsonRecord]:
    documents: list[JsonRecord] = []
    for document in _take(documents_iter, limit):
        doc_id = str(_field(document, "doc_id"))
        text = _document_text(document)
        documents.append({"doc_id": doc_id, "text": text})
    return documents


def _convert_ir_queries(queries_iter: Any, limit: int | None) -> list[JsonRecord]:
    queries: list[JsonRecord] = []
    for query in _take(queries_iter, limit):
        query_id = str(_field(query, "query_id"))
        text = str(_first_available_field(query, ("text", "title", "query")))
        queries.append({"query_id": query_id, "text": text})
    return queries


def _convert_ir_qrels(
    qrels_iter: Any,
    allowed_doc_ids: set[str] | None = None,
    allowed_query_ids: set[str] | None = None,
) -> Qrels:
    qrels: Qrels = {}
    for qrel in qrels_iter:
        query_id = str(_field(qrel, "query_id"))
        doc_id = str(_field(qrel, "doc_id"))
        if allowed_query_ids is not None and query_id not in allowed_query_ids:
            continue
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        relevance = int(_first_available_field(qrel, ("relevance", "iteration")))
        qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels


def _take(items: Any, limit: int | None) -> Any:
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            break
        yield item


def _document_text(document: Any) -> str:
    text = _optional_field(document, "text")
    title = _optional_field(document, "title")
    body = _optional_field(document, "body")
    contents = _optional_field(document, "contents")
    parts = [str(part) for part in (title, text or body or contents) if part]
    if parts:
        return "\n".join(parts)
    return str(_first_available_field(document, ("text", "body", "contents", "title")))


def _first_available_field(item: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = _optional_field(item, name)
        if value is not None:
            return value
    raise ValueError(
        f"Could not find any of fields {names!r} on {type(item).__name__}"
    )


def _field(item: Any, name: str) -> Any:
    value = _optional_field(item, name)
    if value is None:
        raise ValueError(f"Could not find required field {name!r} on {type(item).__name__}")
    return value


def _optional_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
