## Architecture Overview

### Core domains
- Data schemas (`src/schemas/`)
  - `timeseries.py`:
    - `TimeFrame`, `Movement`
    - `IndicatorSeries`, `OHLCVWindow`
    - `QueryBasic`, `QueryTechnical`, `Candidate`
    - `build_candidates_from_dataframe`

- Alpha generation (`src/alphas/`)
  - `momentum.py` → `momentum_reversal`
  - `trend_momentum.py` → `trend_momentum`
  - `rsi_reversion.py` → `rsi_reversion`
  - `volume_price.py` → `volume_price_trend`

- Engines (`src/engines/`)
  - `alpha_engine.py` → `AlphaEngine` and `AlphaSignals`
  - `backtesting_engine.py` → `BacktestingEngine`, `SimulationResults`
  - `portfolio_engine.py` (placeholder for portfolio-level logic if extended)

### StockLLM + FinSeer integration
- Adapters (`src/adapters/`)
  - `alpaca.py`: `symbols_bars_to_df_map`, `to_alpaca_timeframe`
- LLM client (`src/llm/`)
  - `stockllm_client.py`: `StockLLMConfig`, `StockLLMGenerator`
- Retrieval (`src/retrieval/`)
  - `finseer_client.py`: `FinSeerConfig`, `FinSeerEmbedder`
  - `faiss_index.py`: `FaissCandidateIndex`, `default_indicator_builder`, `save/load`
- Orchestration (`src/orchestration/`)
  - `langgraph_pipeline.py`: `build_retrieval_graph`
- Service re-exports (`src/services/__init__.py`) for convenience

### Portfolio construction
- Portfolio (`src/portfolio/optimizer.py`)
  - `PortfolioOptimizer` using PyPortfolioOpt to compute weights from price history (mean-variance, optional L2)

### Retrieval store
- Haystack (`src/retrieval/haystack_store.py`)
  - FAISS-backed `FAISSDocumentStore` wrapper for candidate payloads; pluggable embedder

### API layer
- FastAPI app (`app/`)
  - `schemas.py`:
    - Request/response models: `BuildIndexRequest/Response`, `UpdateIndexRequest/Response`,
      `QueryPredictRequest`/`PredictResponse`, `BacktestRequest/Response`, `PlaceOrderRequest/Response`
  - `main.py`:
    - Endpoints:
      - `POST /index/build`, `POST /index/update` → FAISS index operations
      - `POST /predict` → retrieval + StockLLM prediction over top-k candidates
      - `POST /backtest` → generate alpha signals and run backtest
      - `POST /orders` → placeholder for Alpaca order submission
    - In-memory singletons for FinSeer embedder, FAISS index, and StockLLM generator

### Data flow (typical)
1. Ingest OHLCV bars via API or local DataFrames.
2. Build candidates (5-day windows) per symbol/indicator, serialize to paper JSON.
3. Embed candidates with FinSeer and index in FAISS (with metadata and symbol tracking).
4. At query time:
   - Build `QueryBasic` (5-day adjusted-close) JSON.
   - Retrieve top-k candidates via FAISS (cosine/IP over normalized vectors).
   - Compose prompt: query + candidate payloads; call StockLLM.
   - Parse JSON to movement/probabilities → map to alpha signal.
5. Backtest signals with `BacktestingEngine`.

### External dependencies
- Model runtime: `transformers`, `torch`, `accelerate`, optional `bitsandbytes`.
- Index: `faiss-cpu` (or GPU), numpy fallback if missing.
- Indicators: `talib-binary` (optional; fallback implemented).
- API: `fastapi`, `uvicorn`.
- Optional orchestration: `langgraph`.
- Backtesting/analytics: `pandas`, `numpy`, `tqdm`.

### Notes & extensions
- The FAISS index stores exact candidate JSON payloads to ensure prompt fidelity and fast retrieval.
- Incremental updates add new windows per symbol without re-indexing historical vectors.
- To harden `/predict`, extend the request to include the explicit 5-day OHLCV query window and validate via `QueryBasic`.
- Orders endpoint is a stub; integrate `alpaca-trade-api` and add auth, order status, and position endpoints.


