from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def size_fixed_fractional(weights: Dict[str, float], equity: float) -> Dict[str, float]:
    return {sym: max(0.0, float(w)) * float(equity) for sym, w in weights.items()}


def size_fixed_dollar(weights: Dict[str, float], dollar_per_asset: float) -> Dict[str, float]:
    return {sym: (1.0 if w > 0 else 0.0) * float(dollar_per_asset) for sym, w in weights.items()}


def size_vol_target(
    weights: Dict[str, float], vol_series: pd.Series, risk_budget_per_asset: float
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sym, w in weights.items():
        vol = float(vol_series.get(sym, np.nan))
        if not np.isfinite(vol) or vol <= 0:
            out[sym] = 0.0
        else:
            out[sym] = abs(float(w)) * risk_budget_per_asset / vol
    return out


