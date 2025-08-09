from __future__ import annotations

from dataclasses import dataclass
import logging
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
        logger = logging.getLogger(__name__)
        logger.info(
            "Fetching bars: symbols=%s timeframe=%s start=%s end=%s limit=%s adjustment=%s",
            list(symbols), timeframe, start, end, limit, adjustment,
        )
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
        # Normalize column names and avoid duplicates:
        # - If SDK returns both 'volume' and 'trade_count', keep 'volume' and rename 'trade_count' → 'num_trades'
        # - If only 'trade_count' exists, promote it to 'volume'
        cols = list(bars_df.columns)
        if "trade_count" in cols and "volume" in cols:
            bars_df = bars_df.rename(columns={"trade_count": "num_trades"})
        elif "trade_count" in cols and "volume" not in cols:
            bars_df = bars_df.rename(columns={"trade_count": "volume"})
        # Drop any duplicate-named columns (keep first occurrence)
        if hasattr(bars_df.columns, "duplicated") and bars_df.columns.duplicated().any():
            bars_df = bars_df.loc[:, ~bars_df.columns.duplicated(keep="first")]
        # Normalize timezone handling on the timestamp level for durability.
        # Policy:
        # - For daily/weekly/monthly bars: timestamps are converted to UTC and stored tz-naive
        # - For intraday bars (minute/hour): timestamps are converted to UTC and kept tz-aware
        try:
            ts_level_pos = bars_df.index.names.index("timestamp")
            ts_idx = bars_df.index.levels[ts_level_pos]
            if isinstance(ts_idx, pd.DatetimeIndex):
                is_intraday = timeframe in (
                    TimeFrame.minute,
                    TimeFrame.five_minutes,
                    TimeFrame.fifteen_minutes,
                    TimeFrame.hour,
                )
                if is_intraday:
                    new_ts = ts_idx.tz_localize("UTC") if ts_idx.tz is None else ts_idx.tz_convert("UTC")
                else:
                    # Day/Week/Month → UTC then drop tz
                    if ts_idx.tz is None:
                        new_ts = ts_idx
                    else:
                        new_ts = ts_idx.tz_convert("UTC").tz_localize(None)
                bars_df.index = bars_df.index.set_levels(new_ts, level=ts_level_pos)
                logger.info(
                    "Timezone normalized for bars: intraday=%s tz=%s tz_naive=%s",
                    is_intraday,
                    getattr(new_ts, "tz", None),
                    getattr(new_ts, "tz", None) is None,
                )
        except Exception:
            # If anything unexpected about the index, proceed without normalization
            logger.warning("Failed to normalize timezone for bars index; proceeding as-is")
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
            logger.info("Prepared symbol=%s rows=%d cols=%d", sym, len(df_sym), len(df_sym.columns))
        return out


