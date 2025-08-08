from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

try:
    from alpaca.trading import TradingClient  # type: ignore
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest  # type: ignore
    from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
    _ALPACA_TRADING = True
except Exception:  # pragma: no cover
    _ALPACA_TRADING = False


@dataclass
class AlpacaTradingConfig:
    api_key: str
    secret_key: str
    paper: bool = True


class AlpacaTrading:
    """Simple wrapper for submitting basic orders via alpaca-py TradingClient."""

    def __init__(self, config: AlpacaTradingConfig) -> None:
        if not _ALPACA_TRADING:
            raise RuntimeError("alpaca-py trading not installed. Install `alpaca-py`.")
        self.client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )

    def place_market_order(self, symbol: str, qty: float, side: str, tif: str = "day") -> str:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce[tif.upper()],
        )
        resp = self.client.submit_order(order_data=req)
        return str(resp.id)

    def place_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float, tif: str = "day"
    ) -> str:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce[tif.upper()],
            limit_price=limit_price,
        )
        resp = self.client.submit_order(order_data=req)
        return str(resp.id)

    def get_positions(self) -> Dict[str, float]:
        pos = self.client.get_all_positions()
        out: Dict[str, float] = {}
        for p in pos:
            try:
                out[str(p.symbol)] = float(p.qty)
            except Exception:
                out[str(p.symbol)] = float(getattr(p, "qty", 0))
        return out

    def get_account_equity(self) -> float:
        acc = self.client.get_account()
        try:
            return float(acc.equity)
        except Exception:
            return float(getattr(acc, "equity", 0.0))


