from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns


@dataclass
class PortfolioOptConfig:
    objective: Literal["max_sharpe", "min_vol", "target_return", "target_vol"] = "max_sharpe"
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    risk_model: Literal["sample", "ledoit_wolf", "exp_ewm"] = "sample"
    l2_reg: float = 0.0
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
    # Optional per-asset bounds override
    per_asset_bounds: Optional[Dict[str, Tuple[float, float]]] = None


class PortfolioOptimizer:
    """Wrapper around PyPortfolioOpt supporting multiple objectives and risk models."""

    def __init__(self, config: Optional[PortfolioOptConfig] = None) -> None:
        self.config = config or PortfolioOptConfig()

    def optimize(self, price_df: pd.DataFrame, frequency: int = 252) -> Dict[str, float]:
        # Expected returns
        mu = expected_returns.mean_historical_return(price_df, frequency=frequency)
        # Risk model
        if self.config.risk_model == "sample":
            S = risk_models.sample_cov(price_df, frequency=frequency)
        elif self.config.risk_model == "ledoit_wolf":
            S = risk_models.CovarianceShrinkage(price_df, frequency=frequency).ledoit_wolf()
        else:  # exp_ewm
            rets = price_df.pct_change().dropna()
            ewm = rets.ewm(span=60).cov().dropna()
            # Convert panel-like to covariance matrix (last timestamp)
            last_t = ewm.index.get_level_values(0).max()
            S = ewm.loc[last_t].fillna(0.0)

        # Bounds
        bounds = self.config.weight_bounds
        ef = EfficientFrontier(mu, S, weight_bounds=bounds)
        if self.config.per_asset_bounds:
            ef.add_constraint(lambda w: w)  # no-op to keep EF happy
            for i, sym in enumerate(price_df.columns):
                if sym in self.config.per_asset_bounds:
                    lb, ub = self.config.per_asset_bounds[sym]
                    ef.add_constraint(lambda w, i=i, lb=lb: w[i] >= lb)
                    ef.add_constraint(lambda w, i=i, ub=ub: w[i] <= ub)

        if self.config.l2_reg > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=self.config.l2_reg)

        # Objective
        if self.config.objective == "max_sharpe":
            ef.max_sharpe()
        elif self.config.objective == "min_vol":
            ef.min_volatility()
        elif self.config.objective == "target_return":
            if self.config.target_return is None:
                raise ValueError("target_return must be set for 'target_return' objective")
            ef.efficient_return(self.config.target_return)
        else:  # target_vol
            if self.config.target_volatility is None:
                raise ValueError("target_volatility must be set for 'target_vol' objective")
            ef.efficient_risk(self.config.target_volatility)

        weights = ef.clean_weights()
        return weights


