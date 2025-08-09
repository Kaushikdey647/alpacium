## StockLLM Retrieval and Indexing

This page documents the retrieval/indexing flow used to power StockLLM prompts, including recent improvements for streaming progress and embedder options.

### Components

- FinSeer embedders (`src/retrieval/finseer_client.py`)
  - `FinSeerEmbedder` (HF `AutoModel` + manual pooling)
  - `FinSeerEmbedderST` (SentenceTransformers `ElsaShaw/FinSeer`) – faster for many setups with built-in pooling
- Candidate index (`src/retrieval/faiss_index.py`)
  - `FaissCandidateIndex` (FAISS IP + IDMap2; NumPy fallback)
  - Stores metadata (`symbol`, `candidate_date`, `indicator`) and JSON `payload`

### Payload formats (paper-aligned)

- Query JSON (`QueryBasic.to_paper_json()`):
  - `{"query_stock","query_date","recent_date_list","adjusted_close_list"}`
- Candidate JSON (`Candidate.to_paper_json()`):
  - `{"candidate_stock","candidate_date","recent_date_list","<indicator>_list"}`

### Indexing and progress

`FaissCandidateIndex.add_candidates()` performs batched encoding and inserts into FAISS incrementally. A per-batch progress bar ("Embedding batches") is shown during long runs.

- Embedding normalization occurs once:
  - If the embedder outputs normalized vectors, the index skips re-normalization.
  - Otherwise, the index L2-normalizes prior to insertion for cosine/IP equivalence.

### Persistence

- `save(dir)` writes FAISS index (or NumPy fallback) and JSON metadata.
- `load(dir)` restores the index and in-memory mappings.
- `clear()` removes all vectors and metadata.

### Choosing an embedder

- Use `FinSeerEmbedder` for full HF control (pooling: `cls`, `mean`, `last_token`).
- For speed and simplicity, prefer `FinSeerEmbedderST`:
  - `SentenceTransformer('ElsaShaw/FinSeer')`
  - `encode(texts, batch_size=32, normalize_embeddings=True)`

### Troubleshooting

- Progress appears stuck at 0%: expect a live "Embedding batches" bar; if absent, ensure `show_progress=True`.
- CPU-only runs will be slower for large candidate sets; prefer GPU (`device=cuda`).
- Control candidate volume via smaller `lookback` or a leaner `indicator_builder`.



