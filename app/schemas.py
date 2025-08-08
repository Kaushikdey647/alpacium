from __future__ import annotations

from datetime import datetime, date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

from src.schemas import TimeFrame


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None


class SymbolBars(BaseModel):
    symbol: str
    bars: List[OHLCVBar]


class BuildIndexRequest(BaseModel):
    symbols: List[SymbolBars]
    lookback: int = Field(5, ge=2, le=60)
    timeframe: TimeFrame = TimeFrame.day


class BuildIndexResponse(BaseModel):
    total_candidates: int
    symbols_indexed: List[str]


class UpdateIndexRequest(BaseModel):
    symbols: List[SymbolBars]
    lookback: int = Field(5, ge=2, le=60)
    timeframe: TimeFrame = TimeFrame.day


class UpdateIndexResponse(BaseModel):
    total_added: int
    symbols_updated: List[str]


class QueryPredictRequest(BaseModel):
    symbol: str
    as_of: date
    timeframe: TimeFrame = TimeFrame.day
    top_k: int = Field(5, ge=1, le=50)
    filter_symbols: Optional[List[str]] = None
    # Explicit 5-day window for paper-accurate query composition
    recent_date_list: List[date]
    adjusted_close_list: List[float]

    @validator("adjusted_close_list")
    def _len_match(cls, v, values):
        rdl = values.get("recent_date_list", [])
        if len(v) != len(rdl):
            raise ValueError("adjusted_close_list length must match recent_date_list length")
        if len(v) < 2:
            raise ValueError("Provide at least 2 recent dates for query window (paper uses 5)")
        return v


class CandidateHit(BaseModel):
    id: int
    score: float
    symbol: str
    candidate_date: date
    indicator: str
    payload: str


class PredictResponse(BaseModel):
    movement: Literal["rise", "fall", "freeze"]
    probabilities: Dict[str, float]
    candidates: List[CandidateHit]


class BacktestAlpha(BaseModel):
    name: str
    parameters: Dict[str, float] = Field(default_factory=dict)


class BacktestRequest(BaseModel):
    portfolio_id: Optional[str] = None
    symbols: List[SymbolBars]
    alpha: BacktestAlpha
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.0


class OptimizeRequest(BaseModel):
    # Wide price matrix: list of {symbol, bars} interpreted as a panel to build wide prices
    symbols: List[SymbolBars]
    objective: Literal["max_sharpe", "min_vol", "target_return", "target_vol"] = "max_sharpe"
    risk_model: Literal["sample", "ledoit_wolf", "exp_ewm"] = "sample"
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    l2_reg: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0


class OptimizeResponse(BaseModel):
    weights: Dict[str, float]


class RebalanceRequest(BaseModel):
    portfolio_id: Optional[str] = None
    current_positions: Dict[str, float]
    target_weights: Dict[str, float]
    prices: Dict[str, float]
    equity: float
    lot_size: float = 1.0
    max_pos_percent: float = 0.2


class OrderModel(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    type: Literal["market", "limit"] = "market"
    time_in_force: Literal["day", "gtc"] = "day"
    limit_price: Optional[float] = None


class RebalanceResponse(BaseModel):
    orders: List[OrderModel]


# Portfolio CRUD
class CreatePortfolioRequest(BaseModel):
    name: str
    description: Optional[str] = None
    is_paper: bool = True
    initial_capital: float = 1_000_000.0


class PortfolioModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    is_paper: bool
    initial_capital: float


class BacktestResponse(BaseModel):
    total_pnl: float
    sharpe_ratio: float
    avg_turnover: float


class PlaceOrderRequest(BaseModel):
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    type: Literal["market", "limit"] = "market"
    time_in_force: Literal["day", "gtc", "opg", "ioc", "fok", "cls"] = "day"
    limit_price: Optional[float] = None
    paper: bool = True


class PlaceOrderResponse(BaseModel):
    submitted: bool
    order_id: Optional[str]
    message: Optional[str]


