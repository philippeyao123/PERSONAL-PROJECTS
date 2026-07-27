"""Pre-declared cash, market and simple-strategy benchmarks."""

from __future__ import annotations

import pandas as pd

from systematic_research.exceptions import DataValidationError


def cash_benchmark(index: pd.Index, periodic_rate: float = 0.0) -> pd.Series:
    return pd.Series(periodic_rate, index=index, name="cash")


def market_benchmark(
    returns: pd.DataFrame,
    benchmark_asset: str,
) -> pd.Series:
    required = {"date", "asset", "return"}
    if not required.issubset(returns.columns):
        raise DataValidationError(f"market benchmark requires {sorted(required)}")
    market = returns.loc[returns["asset"] == benchmark_asset].set_index("date")["return"]
    if market.empty:
        raise DataValidationError(f"benchmark asset {benchmark_asset!r} is absent")
    return market.rename("market")


def equal_weight_benchmark(returns: pd.DataFrame) -> pd.Series:
    if not {"date", "asset", "return"}.issubset(returns.columns):
        raise DataValidationError("equal-weight benchmark requires date, asset and return")
    return returns.groupby("date")["return"].mean().rename("equal_weight")
