# NLP Search Benchmark

A clean Python benchmark for comparing lexical, sparse neural, dense, hybrid,
and reranking-based search systems on a shared dataset of queries, documents,
and relevance labels.

## Search Strategies

This project includes six retrieval and ranking strategies:

1. **TF-IDF**: scikit-learn `TfidfVectorizer` with cosine similarity.
2. **BM25**: `rank-bm25` with simple `text.lower().split()` tokenization.
3. **SPLADE**: Hugging Face masked language model encoding into sparse vectors.
4. **Dense Embedding Retrieval**: `sentence-transformers` embeddings with cosine similarity.
5. **Hybrid Retrieval**: BM25 plus dense retrieval with RRF or weighted score fusion.
6. **Cross-Encoder Reranking**: rerank first-stage candidates with a cross-encoder.

Heavy transformer models are lazy-loaded only when their retriever or reranker is
indexed or used.

## Dataset Format

Documents use JSONL:

```json
{ "doc_id": "doc_1", "text": "Document content here." }
```

Queries use JSONL:

```json
{ "query_id": "q_1", "text": "User query here." }
```

Qrels use JSONL:

```json
{ "query_id": "q_1", "doc_id": "doc_1", "relevance": 1 }
```

Relevance labels are:

- `0`: not relevant
- `1`: relevant
- `2`: highly relevant

Sample files are included under `data/processed/` and `data/qrels/`.

## Benchmark Datasets

Local JSONL is the default runtime format for retrieval and evaluation. Larger
benchmark collections can be exported into that format with `ir_datasets`, which
supports BEIR, MS MARCO, and many other IR datasets.

Start with small datasets such as `beir/scifact/test`, and avoid full MS MARCO
exports until the pipeline is validated. MS MARCO exports require `--limit-docs`
unless you explicitly pass `--allow-large-download`.

Validate an existing local JSONL dataset:

```bash
python -m src.pipelines.prepare_dataset \
  --source local \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --qrels data/qrels/sample_qrels.jsonl
```

Export a benchmark dataset to local JSONL:

```bash
python -m src.pipelines.prepare_dataset \
  --source ir_datasets \
  --dataset-name beir/scifact/test \
  --output-dir data/processed/beir_scifact_test \
  --limit-docs 5000 \
  --limit-queries 100
```

This creates:

- `data/processed/beir_scifact_test/documents.jsonl`
- `data/processed/beir_scifact_test/queries.jsonl`
- `data/processed/beir_scifact_test/qrels.jsonl`

Run BM25 on the exported dataset:

```bash
python -m src.pipelines.run_retrieval \
  --method bm25 \
  --documents data/processed/beir_scifact_test/documents.jsonl \
  --queries data/processed/beir_scifact_test/queries.jsonl \
  --output outputs/runs/bm25_scifact.json \
  --top-k 100
```

Evaluate:

```bash
python -m src.pipelines.run_evaluation \
  --qrels data/processed/beir_scifact_test/qrels.jsonl \
  --run outputs/runs/bm25_scifact.json \
  --output outputs/metrics/bm25_scifact.json
```

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Run Retrieval

BM25:

```bash
python -m src.pipelines.run_retrieval \
  --method bm25 \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --output outputs/runs/bm25_sample.json \
  --top-k 5
```

TF-IDF:

```bash
python -m src.pipelines.run_retrieval \
  --method tfidf \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --output outputs/runs/tfidf_sample.json \
  --top-k 5
```

Dense embeddings:

```bash
python -m src.pipelines.run_retrieval \
  --method dense \
  --config configs/dense.yaml \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --output outputs/runs/dense_sample.json \
  --top-k 5
```

SPLADE:

```bash
python -m src.pipelines.run_retrieval \
  --method splade \
  --config configs/splade.yaml \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --output outputs/runs/splade_sample.json \
  --top-k 5
```

Hybrid BM25 plus dense:

```bash
python -m src.pipelines.run_retrieval \
  --method hybrid \
  --config configs/hybrid.yaml \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --output outputs/runs/hybrid_sample.json \
  --top-k 5
```

## Cross-Encoder Reranking

The reranking pipeline expects an existing first-stage run. A common setup is to
retrieve top 100 with BM25, then rerank to top 10.

```bash
python -m src.pipelines.run_reranking \
  --input-run outputs/runs/bm25_sample.json \
  --documents data/processed/sample_documents.jsonl \
  --queries data/processed/sample_queries.jsonl \
  --output outputs/runs/bm25_cross_encoder_sample.json \
  --top-k 10
```

Use `--candidate-k 100` for larger runs.

## Streamlit Demo App

Run the interactive demo app with:

```bash
streamlit run src/app.py
```

Prepare a dataset first with `prepare_dataset.py`. The app looks under
`data/processed/` for folders containing `documents.jsonl`, `queries.jsonl`, and
`qrels.jsonl`, for example `data/processed/beir_scifact_test/`.

In the app you can:

- choose a prepared dataset
- type a manual query or select an existing dataset query
- select retrieval and reranking methods
- view ranked results per method
- expand results to inspect full document text
- compare document ranks and scores across methods

Dense, SPLADE, and cross-encoder models are loaded only when selected. Start
with TF-IDF and BM25 for fast checks, then add heavier methods once the dataset
and query look right.

## Persistent Indexes

Indexes can be built once and reused by the Streamlit app. This avoids rebuilding
retriever indexes every time the app starts or the page reruns.

Build BM25 and TF-IDF indexes:

```bash
python -m src.pipelines.build_indexes \
  --dataset-dir data/processed/beir_scifact_test \
  --methods bm25 tfidf \
  --output-dir outputs/indexes
```

Build a single method:

```bash
python -m src.pipelines.build_indexes \
  --dataset-dir data/processed/beir_scifact_test \
  --methods bm25 \
  --output-dir outputs/indexes
```

Build dense retrieval with a config:

```bash
python -m src.pipelines.build_indexes \
  --dataset-dir data/processed/beir_scifact_test \
  --methods dense \
  --config configs/dense.yaml \
  --output-dir outputs/indexes
```

Saved indexes are stored under:

```text
outputs/indexes/
```

Each saved index includes a `manifest.json` with the dataset name, method,
configuration key, document fingerprint, creation time, and document count. The
app validates indexes with the fingerprint of `documents.jsonl`, so changing the
document file invalidates the old index.

Run the app after building indexes:

```bash
streamlit run src/app.py
```

If a selected method has no valid saved index, the app builds and saves one once,
then reuses it through Streamlit resource caching and future disk loads.

## Adaptive SPLADE-Hybrid Reranking

This project includes an experimental contribution method:

`adaptive_splade_hybrid_rerank`

It combines SPLADE, BM25, and dense retrieval using adaptive weighted Reciprocal
Rank Fusion (RRF), then optionally reranks the fused candidates with a
cross-encoder. The method is designed to keep SPLADE's strong top-ranked
behavior, improve candidate recall with BM25 and dense retrieval, and improve
final ranking with cross-encoder reranking when enabled.

Run:

```bash
python -m src.pipelines.run_retrieval \
  --method adaptive_splade_hybrid_rerank \
  --config configs/adaptive_splade_hybrid_rerank.yaml \
  --documents data/processed/beir_scifact_test/documents.jsonl \
  --queries data/processed/beir_scifact_test/queries.jsonl \
  --output outputs/runs/adaptive_splade_hybrid_rerank_scifact.json \
  --top-k 10
```

Evaluate:

```bash
python -m src.pipelines.run_evaluation \
  --qrels data/processed/beir_scifact_test/qrels.jsonl \
  --run outputs/runs/adaptive_splade_hybrid_rerank_scifact.json \
  --output outputs/metrics/adaptive_splade_hybrid_rerank_scifact.json
```

Tunable parameters in `configs/adaptive_splade_hybrid_rerank.yaml`:

- `splade_top_k`, `bm25_top_k`, `dense_top_k`: candidate depth for each first-stage retriever.
- `candidate_k`: number of fused candidates kept before optional reranking.
- `rrf_k`: reciprocal-rank smoothing constant.
- `fusion.weights`: default SPLADE, BM25, and dense fusion weights.
- `fusion.adaptive_weights`: query-type-specific weights for `entity_heavy`, `long_semantic`, and `default` queries.
- `rerank.enabled`: whether to run cross-encoder reranking.
- `rerank.model_name`: cross-encoder model used for reranking.

## Evaluate

```bash
python -m src.pipelines.run_evaluation \
  --qrels data/qrels/sample_qrels.jsonl \
  --run outputs/runs/bm25_sample.json \
  --output outputs/metrics/bm25_sample_metrics.json
```

The evaluation script writes metrics as JSON. Default cutoffs are `10` and `100`.
Override them with `--k-values`, for example:

```bash
python -m src.pipelines.run_evaluation \
  --qrels data/qrels/sample_qrels.jsonl \
  --run outputs/runs/bm25_sample.json \
  --output outputs/metrics/bm25_sample_metrics.json \
  --k-values 5 10 100
```

## Metrics

- **Precision@K**: fraction of top-K retrieved documents that are relevant.
- **Recall@K**: fraction of all relevant documents retrieved in the top K.
- **MRR@K**: reciprocal rank of the first relevant result within top K.
- **DCG@K**: graded relevance gain discounted by rank.
- **NDCG@K**: DCG normalized by the ideal ranking for the query.

The main reporting metrics are `Recall@10`, `Recall@100`, `MRR@10`,
`NDCG@10`, and `Precision@10`.

## Recommended Experiment Table

| Method               | Recall@10 | Recall@100 | MRR@10 | NDCG@10 | Precision@10 |
| -------------------- | --------: | ---------: | -----: | ------: | -----------: |
| TF-IDF               |           |            |        |         |              |
| BM25                 |           |            |        |         |              |
| Dense Embedding      |           |            |        |         |              |
| SPLADE               |           |            |        |         |              |
| Hybrid BM25 + Dense  |           |            |        |         |              |
| BM25 + Cross-Encoder |           |            |        |         |              |

## Development

Run tests:

```bash
pytest
```

Project modules live under `src/`. All retrievers implement
`BaseRetriever.index()` and `BaseRetriever.search()`, and every result has this
shape:

```json
{
  "doc_id": "doc_1",
  "score": 12.34,
  "text": "document text"
}
```
