"""P&L attribution with reconciliation checks."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from systematic_research.backtest.engine import BacktestResult
from systematic_research.exceptions import DataValidationError


def attribution_by(result: BacktestResult, dimension: str) -> pd.DataFrame:
    """Aggregate gross contribution by asset, sector, signal or another stable dimension."""
    if dimension not in result.positions:
        raise DataValidationError(f"positions do not contain attribution dimension {dimension!r}")
    output = (
        result.positions.groupby(dimension, dropna=False)["gross_contribution"]
        .sum()
        .rename("contribution")
        .reset_index()
    )
    if not np.isclose(output["contribution"].sum(), result.daily["gross_return"].sum(), atol=1e-12):
        raise AssertionError("attribution does not reconcile to gross P&L")
    return output


def period_attribution(result: BacktestResult, frequency: str = "ME") -> pd.DataFrame:
    frame = result.positions.copy()
    frame["period"] = frame["date"].dt.to_period(frequency)
    output = (
        frame.groupby("period")["gross_contribution"].sum().rename("contribution").reset_index()
    )
    if not np.isclose(output["contribution"].sum(), result.daily["gross_return"].sum(), atol=1e-12):
        raise AssertionError("period attribution does not reconcile")
    return output


def signal_attribution(
    signal_contributions: Dict[str, pd.Series],
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(signal_contributions).fillna(0.0)
    if not np.allclose(frame.sum(axis=1), portfolio_returns.reindex(frame.index), atol=1e-12):
        raise AssertionError("signal contributions do not reconcile by period")
    return frame
