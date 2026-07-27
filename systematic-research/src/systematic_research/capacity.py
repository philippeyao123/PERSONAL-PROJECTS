"""Capital participation and performance-capacity curves."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from systematic_research.backtest.costs import SquareRootImpact
from systematic_research.backtest.engine import BacktestResult
from systematic_research.exceptions import DataValidationError


def capacity_curve(
    result: BacktestResult,
    capital_levels: Iterable[float],
    *,
    impact: Optional[SquareRootImpact] = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Reprice historical trades at multiple capital levels."""
    positions = result.positions
    impact_model = impact or SquareRootImpact()
    if not {"adv", "volatility", "absolute_weight_change"}.issubset(positions.columns):
        raise DataValidationError("capacity requires ADV, volatility and historical trades")
    rows = []
    gross = result.daily["gross_return"]
    linear = result.daily["linear_cost"]
    for capital in capital_levels:
        if capital <= 0:
            raise DataValidationError("capital levels must be positive")
        contribution = impact_model.cost(
            positions["absolute_weight_change"],
            positions["volatility"],
            positions["adv"],
            capital,
        )
        impact_daily = contribution.groupby(positions["date"]).sum().reindex(result.daily["date"])
        net = gross.to_numpy() - linear.to_numpy() - impact_daily.to_numpy()
        participation = impact_model.participation(
            positions["absolute_weight_change"], positions["adv"], capital
        )
        rows.append(
            {
                "capital": capital,
                "annualized_net_return": float(np.nanmean(net) * periods_per_year),
                "annualized_impact": float(np.nanmean(impact_daily) * periods_per_year),
                "max_participation": float(participation.max()),
                "p95_participation": float(participation.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values("capital").reset_index(drop=True)


def capacity_limit(curve: pd.DataFrame, max_participation: float = 0.10) -> float:
    feasible = curve.loc[curve["max_participation"] <= max_participation, "capital"]
    return float(feasible.max()) if not feasible.empty else 0.0
