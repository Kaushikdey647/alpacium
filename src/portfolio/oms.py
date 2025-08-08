from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None


def compute_target_shares(
    target_notional: Dict[str, float], prices: Dict[str, float], lot_size: float = 1.0
) -> Dict[str, float]:
    shares: Dict[str, float] = {}
    for sym, notional in target_notional.items():
        p = prices.get(sym)
        if p is None or p <= 0:
            shares[sym] = 0.0
            continue
        qty = (notional / p) if notional > 0 else 0.0
        # round down to lot
        shares[sym] = (qty // lot_size) * lot_size
    return shares


def diff_to_orders(
    current_shares: Dict[str, float], target_shares: Dict[str, float], tif: str = "day"
) -> List[Order]:
    orders: List[Order] = []
    for sym, tgt in target_shares.items():
        cur = float(current_shares.get(sym, 0.0))
        delta = float(tgt) - cur
        if abs(delta) < 1e-6:
            continue
        orders.append(
            Order(symbol=sym, side="buy" if delta > 0 else "sell", qty=abs(delta), time_in_force=tif)
        )
    return orders


def apply_simple_risk_checks(
    orders: List[Order], max_pos_perc: float, equity: float, prices: Dict[str, float]
) -> List[Order]:
    max_notional = max_pos_perc * equity
    filtered: List[Order] = []
    for o in orders:
        p = prices.get(o.symbol, 0.0)
        notional = p * o.qty
        if notional <= max_notional + 1e-6:
            filtered.append(o)
        else:
            # trim qty
            allowed_qty = max(0.0, max_notional / p) if p > 0 else 0.0
            if allowed_qty > 0:
                filtered.append(Order(symbol=o.symbol, side=o.side, qty=allowed_qty, time_in_force=o.time_in_force))
    return filtered


