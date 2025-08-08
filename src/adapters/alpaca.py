from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def symbols_bars_to_df_map(symbols: Iterable[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Convert a list of {symbol, bars[]} dicts to a mapping of symbol -> MultiIndex DataFrame.

    Each bar dict should have keys: timestamp, open, high, low, close, volume, adjusted_close (optional).
    """
    mapping: Dict[str, pd.DataFrame] = {}
    for sb in symbols:
        sym = sb["symbol"]
        bars = sb["bars"]
        rows: List[Dict[str, Any]] = []
        idx: List[Tuple[str, pd.Timestamp]] = []
        for b in bars:
            ts = b["timestamp"]
            ts_dt = pd.to_datetime(ts) if not isinstance(ts, pd.Timestamp) else ts
            idx.append((sym, ts_dt))
            rows.append(
                {
                    "open": float(b["open"]),
                    "high": float(b["high"]),
                    "low": float(b["low"]),
                    "close": float(b["close"]),
                    "volume": float(b["volume"]),
                    "adjusted_close": float(b.get("adjusted_close", b["close"])),
                }
            )
        df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["symbol", "timestamp"]))
        mapping[sym] = df.sort_index()
    return mapping


def to_alpaca_timeframe(value: str | None) -> Any:
    """Best-effort mapping to Alpaca SDK timeframe value (if installed)."""
    try:
        from alpaca.data import TimeFrame as AlpacaTF  # type: ignore

        if value in ("1Min", "1min"):
            return AlpacaTF.Minute
        if value in ("5Min", "5min"):
            return AlpacaTF(5, "Minute")
        if value in ("15Min", "15min"):
            return AlpacaTF(15, "Minute")
        if value in ("1Hour", "1hour"):
            return AlpacaTF.Hour
        if value in ("1Day", "1day"):
            return AlpacaTF.Day
        if value in ("1Week", "1week"):
            return AlpacaTF.Week
        if value in ("1Month", "1month"):
            return AlpacaTF.Month
        return AlpacaTF.Day
    except Exception:
        return value


