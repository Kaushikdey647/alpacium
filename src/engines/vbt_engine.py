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
    sizing: str = "value",  # amount | value | percent | targetpercent
    amount_per_entry: float = 100.0,
    value_per_entry: float = 100_000.0,
    percent_per_entry: float = 0.10,
    cash_sharing: bool = True,
    min_size: float = 1.0,
    size_granularity: float = 1.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    group_by: Optional[str] = None,
    freq: str = "1D",
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

    if sizing.lower() == "targetpercent":
        # Build target weights: +w for long, -w for short; normalize by active counts
        long_w = (entries.replace(False, np.nan)).ffill().where(sig_w > 0, other=np.nan)
        short_w = (short_entries.replace(False, np.nan)).ffill().where(sig_w < 0, other=np.nan)
        n_long = (sig_w > 0).sum(axis=1).replace(0, np.nan)
        n_short = (sig_w < 0).sum(axis=1).replace(0, np.nan)
        w_long = (1.0 * (sig_w > 0)).div(n_long, axis=0).fillna(0.0)
        w_short = (-1.0 * (sig_w < 0)).div(n_short, axis=0).fillna(0.0)
        target_w = (w_long + w_short).fillna(0.0)

        pf = vbt.Portfolio.from_orders(
            close=close_w,
            size=target_w,
            size_type="targetpercent",
            init_cash=initial_capital,
            cash_sharing=cash_sharing,
            fees=transaction_cost,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            size_granularity=size_granularity,
            group_by=group_by,
            freq=freq,
        )
    else:
        # Construct per-entry size field
        if sizing.lower() == "amount":
            size_const = pd.DataFrame(amount_per_entry, index=close_w.index, columns=close_w.columns)
            size = size_const.where(entries | short_entries, other=0.0)
            size_type = "amount"
        elif sizing.lower() == "percent":
            size_const = pd.DataFrame(percent_per_entry, index=close_w.index, columns=close_w.columns)
            size = size_const.where(entries | short_entries, other=0.0)
            size_type = "percent"
        else:  # value (default)
            size_const = pd.DataFrame(value_per_entry, index=close_w.index, columns=close_w.columns)
            size = size_const.where(entries | short_entries, other=0.0)
            size_type = "value"

        pf = vbt.Portfolio.from_signals(
            close=close_w,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=initial_capital,
            fees=transaction_cost,
            fixed_fees=fixed_fees,
            slippage=slippage,
            size=size,
            size_type=size_type,
            cash_sharing=cash_sharing,
            min_size=min_size,
            size_granularity=size_granularity,
            freq=freq,
        )

    stats = pf.stats()
    total_return = (
        float(stats.loc["Total Return [%]"]) / 100.0
        if "Total Return [%]" in stats.index
        else float(pf.total_return())
    )
    # vectorbt labels this as "Annualized Return [%]"; fallback to method for compatibility
    if "Annualized Return [%]" in stats.index:
        annual_return = float(stats.loc["Annualized Return [%]"]) / 100.0
    elif "Annual Return [%]" in stats.index:
        # Backward compatibility if some env uses older label
        annual_return = float(stats.loc["Annual Return [%]"]) / 100.0
    else:
        # Method name is annualized_return in recent versions
        annual_return = float(getattr(pf, "annualized_return")())

    sharpe = (
        float(stats.loc["Sharpe Ratio"]) if "Sharpe Ratio" in stats.index else float(pf.sharpe_ratio())
    )

    return VBTResults(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        portfolio=pf,
    )


