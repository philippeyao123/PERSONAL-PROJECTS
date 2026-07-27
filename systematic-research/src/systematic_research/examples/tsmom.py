"""Compact time-series momentum replication building block."""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def time_series_momentum_positions(
    returns: pd.DataFrame,
    *,
    lookback: int = 252,
    volatility_window: int = 60,
    target_volatility: float = 0.10,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Past-only sign momentum with inverse-volatility scaling."""
    if min(lookback, volatility_window, periods_per_year) <= 1 or target_volatility <= 0:
        raise DataValidationError("TSMOM parameters are invalid")
    ordered = returns.sort_values(["asset", "date"]).copy()
    grouped = ordered.groupby("asset", sort=False)["return"]
    cumulative = grouped.transform(
        lambda values: (1.0 + values).shift(1).rolling(lookback).apply(np.prod, raw=True) - 1.0
    )
    volatility = grouped.transform(
        lambda values: (
            values.shift(1).rolling(volatility_window).std(ddof=1) * np.sqrt(periods_per_year)
        )
    )
    ordered["target_weight"] = (np.sign(cumulative) * target_volatility / volatility).replace(
        [np.inf, -np.inf], np.nan
    )
    ordered["available_at"] = ordered["date"]
    return ordered.dropna(subset=["target_weight"])
