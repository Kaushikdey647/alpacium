from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import vectorbt as vbt  # type: ignore

from src.engines.alpha_engine import AlphaSignals


@dataclass
class VBTResults:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    portfolio: "vbt.portfolio.base.Portfolio"  # type: ignore


def _prepare_wide_frames(
    historical_data: pd.DataFrame, signals_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_w = historical_data["close"].unstack(level=0).sort_index()
    sig_w = signals_df["signal"].unstack(level=0).reindex(close_w.index).fillna(0.0)
    return close_w, sig_w


def run_backtest_vbt(
    historical_data: pd.DataFrame,
    alpha_signals: Union[pd.DataFrame, AlphaSignals],
    initial_capital: float = 1_000_000.0,
    transaction_cost: float = 0.001,
) -> VBTResults:
    signals_df = alpha_signals.signals if isinstance(alpha_signals, AlphaSignals) else alpha_signals
    close_w, sig_w = _prepare_wide_frames(historical_data, signals_df)

    # Generate entries/exits for longs and shorts on sign changes
    long_now = sig_w > 0
    long_prev = long_now.shift(1, fill_value=False)
    entries = long_now & ~long_prev
    exits = ~long_now & long_prev

    short_now = sig_w < 0
    short_prev = short_now.shift(1, fill_value=False)
    short_entries = short_now & ~short_prev
    short_exits = ~short_now & short_prev

    # Allocate per-symbol budget for entries
    n_symbols = close_w.shape[1]
    alloc_per_symbol = initial_capital / max(n_symbols, 1)
    size_long = (alloc_per_symbol / close_w).where(entries, other=0.0)
    size_short = (alloc_per_symbol / close_w).where(short_entries, other=0.0)

    pf = vbt.Portfolio.from_signals(
        close=close_w,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=initial_capital,
        fees=transaction_cost,
        size=size_long,
        short_size=size_short,
        freq="1D",
    )

    stats = pf.stats()
    total_return = float(stats.loc["Total Return [%]"]) / 100.0 if "Total Return [%]" in stats.index else float(pf.total_return())
    annual_return = float(stats.loc["Annual Return [%]"]) / 100.0 if "Annual Return [%]" in stats.index else float(pf.annual_return())
    sharpe = float(stats.loc["Sharpe Ratio"]) if "Sharpe Ratio" in stats.index else float(pf.sharpe_ratio())

    return VBTResults(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        portfolio=pf,
    )


