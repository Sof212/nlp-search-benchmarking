"""Streamlit app for interactive search benchmark exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dataset import load_documents, load_qrels, load_queries, queries_by_id
from src.evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.index_store import compute_file_fingerprint
from src.pipelines.build_indexes import (
    IndexedRetrieverResult,
    get_index_status,
    load_or_build_retriever_index,
)
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.retrievers.adaptive_splade_hybrid import AdaptiveSpladeHybridRetriever
from src.utils import load_yaml_config

DATA_ROOT = Path("data/processed")
INDEX_ROOT = Path("outputs/indexes")
REQUIRED_DATASET_FILES = ("documents.jsonl", "queries.jsonl", "qrels.jsonl")
TEXT_PREVIEW_CHARS = 300

METHOD_LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "dense": "Dense Embedding",
    "splade": "SPLADE",
    "hybrid": "Hybrid BM25 + Dense",
    "cross_encoder": "BM25 + Cross-Encoder",
    "adaptive_splade_hybrid_rerank": "Adaptive SPLADE Hybrid Rerank",
}
DEFAULT_METHODS = [
    "TF-IDF",
    "BM25",
    "Dense Embedding",
    "Hybrid BM25 + Dense",
]
COMPARISON_METHODS = [
    "tfidf",
    "bm25",
    "dense",
    "splade",
    "hybrid",
    "cross_encoder",
    "adaptive_splade_hybrid_rerank",
]


def main() -> None:
    st.set_page_config(page_title="NLP Search Benchmark", layout="wide")
    st.title("NLP Search Benchmark")

    dataset_dirs = discover_datasets()
    if not dataset_dirs:
        st.info("No prepared dataset found. Run prepare_dataset.py first.")
        return

    selected_dataset = st.selectbox(
        "Prepared dataset",
        options=dataset_dirs,
        format_func=lambda path: str(Path(path).relative_to(DATA_ROOT))
        if Path(path).is_relative_to(DATA_ROOT)
        else str(path),
    )

    try:
        documents_fingerprint = compute_file_fingerprint(
            Path(selected_dataset) / "documents.jsonl"
        )
        documents, queries, qrels = load_selected_dataset(
            selected_dataset,
            documents_fingerprint,
        )
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    top_k_retrieval, top_k_rerank_candidates, device_choice = render_sidebar()
    query_id, query_text, has_labels = render_query_selector(queries, qrels)
    selected_methods = render_method_selector()

    if st.button("Run search", type="primary"):
        run_search(
            dataset_dir=selected_dataset,
            query_text=query_text,
            query_id=query_id,
            qrels=qrels,
            selected_methods=selected_methods,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank_candidates=top_k_rerank_candidates,
            device_choice=device_choice,
            has_labels=has_labels,
        )


@st.cache_data(show_spinner=False)
def discover_datasets(data_root: str = str(DATA_ROOT)) -> list[str]:
    """Find prepared dataset directories under ``data/processed``."""
    root = Path(data_root)
    if not root.exists():
        return []
    dataset_dirs: list[str] = []
    for documents_path in root.rglob("documents.jsonl"):
        dataset_dir = documents_path.parent
        if all((dataset_dir / filename).exists() for filename in REQUIRED_DATASET_FILES):
            dataset_dirs.append(str(dataset_dir))
    return sorted(set(dataset_dirs))


@st.cache_data(show_spinner="Loading dataset...")
def load_selected_dataset(
    dataset_dir: str,
    documents_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Load documents, queries, and qrels from a prepared dataset directory."""
    _ = documents_fingerprint
    root = Path(dataset_dir)
    return (
        load_documents(root / "documents.jsonl"),
        load_queries(root / "queries.jsonl"),
        load_qrels(root / "qrels.jsonl"),
    )


def render_sidebar() -> tuple[int, int, str]:
    """Render retrieval controls in the sidebar."""
    st.sidebar.header("Retrieval settings")
    top_k_retrieval = st.sidebar.number_input(
        "top_k_retrieval",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )
    top_k_rerank_candidates = st.sidebar.number_input(
        "top_k_rerank_candidates",
        min_value=10,
        max_value=200,
        value=100,
        step=10,
    )
    device_choice = st.sidebar.selectbox("device", options=["auto", "cpu", "cuda"])
    return int(top_k_retrieval), int(top_k_rerank_candidates), str(device_choice)


def render_query_selector(
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
) -> tuple[str | None, str, bool]:
    """Render manual and dataset query controls."""
    query_mode = st.radio(
        "Query mode",
        options=["Manual query", "Existing dataset query"],
        horizontal=True,
    )
    if query_mode == "Manual query":
        query_text = st.text_input("Enter your query")
        st.caption("No relevance labels available for manual queries.")
        return None, query_text.strip(), False

    indexed_queries = queries_by_id(queries)
    query_ids = sorted(indexed_queries)
    if not query_ids:
        st.warning("This dataset has no queries.")
        return None, "", False
    selected_query_id = st.selectbox(
        "Dataset query",
        options=query_ids,
        format_func=lambda query_id: _query_option_label(indexed_queries[query_id]),
    )
    query_text = str(indexed_queries[selected_query_id]["text"])
    has_labels = selected_query_id in qrels
    if not has_labels:
        st.caption("No relevance labels found for this dataset query.")
    return selected_query_id, query_text, has_labels


def render_method_selector() -> list[str]:
    """Render retrieval and reranking method selection."""
    label_to_key = {label: key for key, label in METHOD_LABELS.items()}
    selected_labels = st.multiselect(
        "Methods",
        options=list(METHOD_LABELS.values()),
        default=DEFAULT_METHODS,
    )
    return [label_to_key[label] for label in selected_labels]


def run_search(
    dataset_dir: str,
    query_text: str,
    query_id: str | None,
    qrels: dict[str, dict[str, int]],
    selected_methods: list[str],
    top_k_retrieval: int,
    top_k_rerank_candidates: int,
    device_choice: str,
    has_labels: bool,
) -> None:
    """Run selected methods and render results."""
    if not query_text:
        st.warning("Enter a query before running search.")
        return
    if not selected_methods:
        st.warning("Select at least one method.")
        return

    all_results: dict[str, list[dict[str, Any]]] = {}
    query_qrels = qrels.get(query_id or "", {}) if has_labels else {}

    for method_key in selected_methods:
        with st.spinner(f"Running {METHOD_LABELS[method_key]}..."):
            try:
                if method_key == "cross_encoder":
                    results = run_cross_encoder_search(
                        dataset_dir=dataset_dir,
                        query_text=query_text,
                        top_k_retrieval=top_k_retrieval,
                        top_k_rerank_candidates=top_k_rerank_candidates,
                        device_choice=device_choice,
                    )
                elif method_key == "adaptive_splade_hybrid_rerank":
                    st.warning(
                        "Adaptive SPLADE Hybrid Rerank loads SPLADE, dense, "
                        "and cross-encoder models when reranking is enabled."
                    )
                    documents_fingerprint = compute_file_fingerprint(
                        Path(dataset_dir) / "documents.jsonl"
                    )
                    retriever = build_adaptive_retriever(
                        dataset_dir,
                        device_choice,
                        documents_fingerprint,
                    )
                    results = retriever.search(query_text, top_k=top_k_retrieval)
                else:
                    documents_fingerprint = compute_file_fingerprint(
                        Path(dataset_dir) / "documents.jsonl"
                    )
                    valid_index, index_dir = get_index_status(
                        method=method_key,
                        dataset_dir=dataset_dir,
                        output_dir=INDEX_ROOT,
                        device=resolve_device(device_choice)
                        if method_key in {"dense", "splade", "hybrid"}
                        else None,
                    )
                    if valid_index:
                        st.caption(
                            f"{METHOD_LABELS[method_key]}: loading saved index "
                            f"from {index_dir}"
                        )
                    else:
                        st.caption(
                            f"{METHOD_LABELS[method_key]}: no valid saved index; "
                            "building and saving one now"
                        )
                    index_result = build_retriever(
                        method_key,
                        dataset_dir,
                        device_choice,
                        documents_fingerprint,
                    )
                    st.caption(
                        f"{METHOD_LABELS[method_key]}: {index_result.status} index "
                        f"at {index_result.index_dir}"
                    )
                    retriever = index_result.retriever
                    results = retriever.search(query_text, top_k=top_k_retrieval)
            except MemoryError:
                st.error(
                    f"{METHOD_LABELS[method_key]} ran out of memory. "
                    "Try CPU, a smaller dataset, or fewer candidates."
                )
                continue
            except ImportError as exc:
                st.error(str(exc))
                continue
            except ValueError as exc:
                st.error(str(exc))
                continue
            except Exception as exc:
                st.error(f"{METHOD_LABELS[method_key]} failed: {exc}")
                continue
        all_results[method_key] = results

    if not all_results:
        st.warning("No results to display.")
        return

    render_results(all_results, query_qrels)
    render_comparison(all_results, query_qrels if has_labels else None)
    if has_labels:
        render_single_query_metrics(all_results, query_qrels, top_k_retrieval)
    else:
        st.info("No relevance labels available for manual queries.")


@st.cache_resource(show_spinner=False)
def build_retriever(
    method_key: str,
    dataset_dir: str,
    device_choice: str,
    documents_fingerprint: str,
) -> IndexedRetrieverResult:
    """Load or build a cached persistent retriever for a selected dataset."""
    _ = documents_fingerprint
    device = (
        resolve_device(device_choice)
        if method_key in {"dense", "splade", "hybrid"}
        else None
    )
    return load_or_build_retriever_index(
        method=method_key,
        dataset_dir=dataset_dir,
        output_dir=INDEX_ROOT,
        device=device,
    )


@st.cache_resource(show_spinner=False)
def build_cross_encoder(device_choice: str) -> CrossEncoderReranker:
    """Build a cached cross-encoder reranker."""
    return CrossEncoderReranker(device=resolve_device(device_choice))


def run_cross_encoder_search(
    dataset_dir: str,
    query_text: str,
    top_k_retrieval: int,
    top_k_rerank_candidates: int,
    device_choice: str,
) -> list[dict[str, Any]]:
    """Retrieve with BM25, then rerank candidates with the cross-encoder."""
    documents_fingerprint = compute_file_fingerprint(Path(dataset_dir) / "documents.jsonl")
    bm25 = build_retriever("bm25", dataset_dir, "auto", documents_fingerprint).retriever
    candidates = bm25.search(
        query_text,
        top_k=max(top_k_retrieval, top_k_rerank_candidates),
    )
    reranker = build_cross_encoder(device_choice)
    return reranker.rerank(query_text, candidates, top_k=top_k_retrieval)


@st.cache_resource(show_spinner=False)
def build_adaptive_retriever(
    dataset_dir: str,
    device_choice: str,
    documents_fingerprint: str,
) -> AdaptiveSpladeHybridRetriever:
    """Build the adaptive contribution method from cached component indexes."""
    _ = documents_fingerprint
    config = load_yaml_config("configs/adaptive_splade_hybrid_rerank.yaml")
    device = resolve_device(device_choice)
    splade_config = {"retriever": dict(config.get("splade", {}))}
    bm25_config = {"retriever": dict(config.get("bm25", {}))}
    dense_config = {"retriever": dict(config.get("dense", {}))}
    bm25_result = load_or_build_retriever_index(
        method="bm25",
        dataset_dir=dataset_dir,
        output_dir=INDEX_ROOT,
        config=bm25_config,
    )
    dense_result = load_or_build_retriever_index(
        method="dense",
        dataset_dir=dataset_dir,
        output_dir=INDEX_ROOT,
        config=dense_config,
        device=device,
    )
    splade_result = load_or_build_retriever_index(
        method="splade",
        dataset_dir=dataset_dir,
        output_dir=INDEX_ROOT,
        config=splade_config,
        device=device,
    )
    return AdaptiveSpladeHybridRetriever.from_config(
        config,
        splade_retriever=splade_result.retriever,
        bm25_retriever=bm25_result.retriever,
        dense_retriever=dense_result.retriever,
        device=device,
    )


def resolve_device(device_choice: str) -> str | None:
    """Resolve UI device selection into model constructor arguments."""
    if device_choice == "auto":
        return None
    if device_choice == "cpu":
        return "cpu"
    if device_choice == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch before using CUDA models") from exc
        if not torch.cuda.is_available():
            raise ValueError("CUDA was selected but is not available on this machine")
        return "cuda"
    raise ValueError(f"Unsupported device choice: {device_choice}")


def render_results(
    all_results: dict[str, list[dict[str, Any]]],
    query_qrels: dict[str, int],
) -> None:
    """Render per-method result tables and expandable full text."""
    st.subheader("Results")
    tabs = st.tabs([METHOD_LABELS[method_key] for method_key in all_results])
    for tab, (method_key, results) in zip(tabs, all_results.items()):
        with tab:
            st.dataframe(
                format_results(results, query_qrels=query_qrels),
                use_container_width=True,
                hide_index=True,
            )
            for rank, result in enumerate(results, start=1):
                doc_id = str(result["doc_id"])
                with st.expander(f"{rank}. {doc_id}"):
                    st.write(str(result.get("text", "")))


def format_results(
    results: list[dict[str, Any]],
    query_qrels: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Format ranked results for display."""
    rows: list[dict[str, Any]] = []
    include_relevance = query_qrels is not None and bool(query_qrels)
    for rank, result in enumerate(results, start=1):
        doc_id = str(result["doc_id"])
        row: dict[str, Any] = {
            "rank": rank,
            "doc_id": doc_id,
            "score": float(result.get("score", 0.0)),
        }
        if include_relevance:
            row["relevance"] = query_qrels.get(doc_id, 0)
        row["text"] = _truncate_text(str(result.get("text", "")))
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison(
    all_results: dict[str, list[dict[str, Any]]],
    query_qrels: dict[str, int] | None,
) -> None:
    """Render a combined rank and score comparison table."""
    st.subheader("Comparison")
    comparison = build_comparison_table(all_results, query_qrels=query_qrels)
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def build_comparison_table(
    all_results: dict[str, list[dict[str, Any]]],
    query_qrels: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build a combined comparison table across methods."""
    by_doc: dict[str, dict[str, Any]] = {}
    for method_key, results in all_results.items():
        for rank, result in enumerate(results, start=1):
            doc_id = str(result["doc_id"])
            row = by_doc.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "text_preview": _truncate_text(str(result.get("text", ""))),
                    "_best_rank": rank,
                },
            )
            row["_best_rank"] = min(int(row["_best_rank"]), rank)
            row[f"{method_key}_rank"] = rank
            row[f"{method_key}_score"] = float(result.get("score", 0.0))

    rows = sorted(by_doc.values(), key=lambda row: (row["_best_rank"], row["doc_id"]))
    include_relevance = query_qrels is not None and bool(query_qrels)
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {
            "doc_id": row["doc_id"],
            "text_preview": row["text_preview"],
        }
        if include_relevance:
            output["relevance"] = query_qrels.get(str(row["doc_id"]), 0)
        for method_key in COMPARISON_METHODS:
            output[f"{method_key}_rank"] = row.get(f"{method_key}_rank")
            output[f"{method_key}_score"] = row.get(f"{method_key}_score")
        table_rows.append(output)
    return pd.DataFrame(table_rows)


def render_single_query_metrics(
    all_results: dict[str, list[dict[str, Any]]],
    query_qrels: dict[str, int],
    k: int,
) -> None:
    """Render single-query metrics when relevance labels are available."""
    if not query_qrels:
        st.info("No relevance labels found for this dataset query.")
        return
    rows = []
    for method_key, results in all_results.items():
        rows.append(
            {
                "method": METHOD_LABELS[method_key],
                f"Precision@{k}": precision_at_k(query_qrels, results, k),
                f"Recall@{k}": recall_at_k(query_qrels, results, k),
                f"MRR@{k}": mrr_at_k(query_qrels, results, k),
                f"NDCG@{k}": ndcg_at_k(query_qrels, results, k),
            }
        )
    st.subheader("Single-query metrics")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _query_option_label(query: dict[str, Any]) -> str:
    return f"{query['query_id']} | {_truncate_text(str(query['text']), 100)}"


def _truncate_text(text: str, max_chars: int = TEXT_PREVIEW_CHARS) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return f"{clean[: max_chars - 3]}..."


if __name__ == "__main__":
    main()
