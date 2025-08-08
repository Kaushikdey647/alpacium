from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:  # alpaca-py (preferred)
    from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
    from alpaca.data.requests import StockBarsRequest  # type: ignore
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame  # type: ignore
    _ALPACA_PY = True
except Exception:  # pragma: no cover
    _ALPACA_PY = False

from src.schemas import TimeFrame


def _to_alpaca_timeframe(tf: TimeFrame) -> "AlpacaTimeFrame | str":
    if not _ALPACA_PY:
        return tf.value
    mapping = {
        TimeFrame.minute: AlpacaTimeFrame.Minute,
        TimeFrame.five_minutes: AlpacaTimeFrame(5, "Minute"),
        TimeFrame.fifteen_minutes: AlpacaTimeFrame(15, "Minute"),
        TimeFrame.hour: AlpacaTimeFrame.Hour,
        TimeFrame.day: AlpacaTimeFrame.Day,
        TimeFrame.week: AlpacaTimeFrame.Week,
        TimeFrame.month: AlpacaTimeFrame.Month,
    }
    return mapping.get(tf, AlpacaTimeFrame.Day)


@dataclass
class AlpacaMarketDataConfig:
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    use_paper: bool = True


class AlpacaMarketData:
    """Adapter around alpaca-py StockHistoricalDataClient to fetch bars.

    Returns symbol->DataFrame mappings with MultiIndex (symbol, timestamp)
    and columns: open, high, low, close, adjusted_close, volume.
    """

    def __init__(self, config: Optional[AlpacaMarketDataConfig] = None) -> None:
        if not _ALPACA_PY:
            raise RuntimeError("alpaca-py not installed. Install `alpaca-py`.\n")
        self.config = config or AlpacaMarketDataConfig()
        # For data client, keys can be passed via env for stocks
        self.client = StockHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key,
        )

    def get_stock_bars(
        self,
        symbols: Iterable[str],
        timeframe: TimeFrame = TimeFrame.day,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical bars and return symbol->DataFrame mapping.

        The DataFrames have MultiIndex (symbol, timestamp) and numeric columns.
        """
        req = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=_to_alpaca_timeframe(timeframe),
            start=start,
            end=end,
            limit=limit,
            adjustment=adjustment,
        )
        result = self.client.get_stock_bars(req)
        # alpaca-py returns a DataFrame-like object with `.df`
        bars_df = getattr(result, "df", result)
        if not isinstance(bars_df, pd.DataFrame):
            raise RuntimeError("Unexpected response type from Alpaca get_stock_bars")
        # Expect index with columns [symbol, timestamp]
        if not isinstance(bars_df.index, pd.MultiIndex):
            # Try to set MultiIndex if symbol is a column
            if {"symbol", "timestamp"}.issubset(set(bars_df.columns)):
                bars_df = bars_df.set_index(["symbol", "timestamp"]).sort_index()
            else:
                raise RuntimeError("Bars DataFrame missing MultiIndex or symbol/timestamp columns")
        bars_df = bars_df.rename(columns={"trade_count": "volume"})
        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in bars_df.columns:
                bars_df[col] = pd.NA
        bars_df["adjusted_close"] = bars_df.get("adjusted_close", bars_df["close"]).astype(float)
        # Split into symbol dict
        out: Dict[str, pd.DataFrame] = {}
        for sym in bars_df.index.get_level_values(0).unique():
            df_sym = bars_df.loc[sym][["open", "high", "low", "close", "adjusted_close", "volume"]].astype(float)
            df_sym.index.name = "timestamp"
            out[str(sym)] = df_sym
        return out


