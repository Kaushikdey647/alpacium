from __future__ import annotations

import os
from datetime import date
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import BacktestRequest, BacktestResponse, PlaceOrderRequest, PlaceOrderResponse
from src.engines.alpha_engine import AlphaEngine
from src.engines.vbt_engine import run_backtest_vbt
from src.schemas import TimeFrame
from src.adapters.alpaca import symbols_bars_to_df_map
from src.adapters.alpaca_data import AlpacaMarketData, AlpacaMarketDataConfig
from src.adapters.alpaca_trading import AlpacaTrading, AlpacaTradingConfig
from src.retrieval.faiss_index import FaissCandidateIndex, default_indicator_builder
from src.retrieval.finseer_client import FinSeerEmbedder, FinSeerConfig
from src.llm.stockllm_client import StockLLMGenerator, StockLLMConfig


app = FastAPI(title="Alpacium Trading API", version="0.1.0")


# --- Singletons (simple in-memory for now) -------------------------------------

_embedder = FinSeerEmbedder(FinSeerConfig())
_index = FaissCandidateIndex(_embedder)
_generator = StockLLMGenerator(StockLLMConfig())
_market = None  # Lazy
_trader = None  # Lazy
_db = None  # Lazy Supabase


def _symbols_to_df_map(symbols: List[dict]) -> Dict[str, pd.DataFrame]:
    return symbols_bars_to_df_map(symbols)


from app.routes.indexing import router as indexing_router
app.include_router(indexing_router)


from app.routes.predict import router as predict_router
app.include_router(predict_router)


@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    # Convert incoming bars to DataFrame
    sym_map = _symbols_to_df_map([s.dict() for s in req.symbols])
    df = pd.concat(sym_map.values()).sort_index()

    # Select alpha function
    from src.alphas import momentum_reversal, trend_momentum, rsi_reversion, volume_price_trend
    alpha_map = {
        "momentum_reversal": momentum_reversal,
        "trend_momentum": trend_momentum,
        "rsi_reversion": rsi_reversion,
        "volume_price_trend": volume_price_trend,
    }
    fn = alpha_map.get(req.alpha.name)
    if not fn:
        raise HTTPException(status_code=400, detail=f"Unknown alpha {req.alpha.name}")

    # Generate signals
    engine = AlphaEngine()
    alpha_signals = engine.generate_signals(df, fn, parameters=req.alpha.parameters)

    vbt_res = run_backtest_vbt(
        historical_data=df,
        alpha_signals=alpha_signals,
        initial_capital=req.initial_capital,
        transaction_cost=req.transaction_cost,
    )
    # Map vectorbt stats to response (avg_turnover omitted for now)
    return BacktestResponse(
        total_pnl=float(vbt_res.total_return * req.initial_capital),
        sharpe_ratio=float(vbt_res.sharpe_ratio),
        avg_turnover=0.0,
    )


from app.routes.portfolio import router as portfolio_router
app.include_router(portfolio_router)
from app.routes.embeddings import router as embeddings_router
app.include_router(embeddings_router)
from app.routes.bars import router as bars_router
app.include_router(bars_router)


@app.post("/orders", response_model=PlaceOrderResponse)
def place_order(req: PlaceOrderRequest) -> PlaceOrderResponse:
    try:
        global _trader
        if _trader is None:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                return PlaceOrderResponse(submitted=False, order_id=None, message="Missing ALPACA_API_KEY/SECRET_KEY")
            _trader = AlpacaTrading(AlpacaTradingConfig(api_key=api_key, secret_key=secret_key, paper=req.paper))
        if req.type == "market":
            oid = _trader.place_market_order(req.symbol, req.qty, req.side, req.time_in_force)
        else:
            if req.limit_price is None:
                return PlaceOrderResponse(submitted=False, order_id=None, message="limit_price required for limit orders")
            oid = _trader.place_limit_order(req.symbol, req.qty, req.side, req.limit_price, req.time_in_force)
        return PlaceOrderResponse(submitted=True, order_id=oid, message=None)
    except Exception as e:
        return PlaceOrderResponse(submitted=False, order_id=None, message=str(e))


from app.routes.portfolios_crud import router as portfolios_router
app.include_router(portfolios_router)


