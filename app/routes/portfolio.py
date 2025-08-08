from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from app.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    RebalanceRequest,
    RebalanceResponse,
)
from app.deps import get_db
from src.adapters.alpaca import symbols_bars_to_df_map
from src.portfolio.optimizer import PortfolioOptimizer, PortfolioOptConfig
from src.portfolio.sizing import size_fixed_fractional
from src.portfolio.oms import compute_target_shares, diff_to_orders, apply_simple_risk_checks


router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    sym_map = symbols_bars_to_df_map([s.dict() for s in req.symbols])
    price_wide = pd.concat({sym: df["close"] for sym, df in sym_map.items()}, axis=1).dropna(how="all")
    cfg = PortfolioOptConfig(
        objective=req.objective,
        target_return=req.target_return,
        target_volatility=req.target_volatility,
        risk_model=req.risk_model,
        l2_reg=req.l2_reg,
        weight_bounds=(req.lower_bound, req.upper_bound),
    )
    opt = PortfolioOptimizer(cfg)
    weights = opt.optimize(price_wide)
    return OptimizeResponse(weights=weights)


@router.post("/rebalance", response_model=RebalanceResponse)
def rebalance(req: RebalanceRequest) -> RebalanceResponse:
    target_notional = size_fixed_fractional(req.target_weights, req.equity)
    tgt_shares = compute_target_shares(target_notional, req.prices, lot_size=req.lot_size)
    orders = diff_to_orders(req.current_positions, tgt_shares, tif="day")
    orders = apply_simple_risk_checks(orders, max_pos_perc=req.max_pos_percent, equity=req.equity, prices=req.prices)
    return RebalanceResponse(orders=[
        dict(symbol=o.symbol, side=o.side, qty=o.qty, type=o.type, time_in_force=o.time_in_force, limit_price=o.limit_price)
    for o in orders])


