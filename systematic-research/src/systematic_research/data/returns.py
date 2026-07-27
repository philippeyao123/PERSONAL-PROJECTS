"""Return transformations with explicit conventions."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def compute_returns(
    frame: pd.DataFrame,
    kind: Literal["simple", "log"] = "simple",
    *,
    price_column: str = "price",
) -> pd.Series:
    """Compute per-asset close-to-close returns without cross-asset filling."""
    if not {"date", "asset", price_column}.issubset(frame.columns):
        raise DataValidationError("returns require date, asset and price columns")
    ordered = frame.sort_values(["asset", "date"], kind="stable")
    grouped = ordered.groupby("asset", sort=False)[price_column]
    if kind == "simple":
        result = grouped.pct_change(fill_method=None)
    elif kind == "log":
        result = grouped.transform(lambda values: np.log(values).diff())
    else:
        raise DataValidationError("return kind must be 'simple' or 'log'")
    result.index = ordered.index
    return result.sort_index()


def excess_returns(returns: pd.Series, risk_free: Union[pd.Series, float]) -> pd.Series:
    """Subtract a frequency-matched risk-free return."""
    return returns - risk_free


def roll_adjusted_futures_returns(
    frame: pd.DataFrame,
    *,
    contract_column: str = "contract",
    price_column: str = "price",
) -> pd.Series:
    """Compute returns within contracts so rolls never appear as economic returns."""
    required = {"date", "asset", contract_column, price_column}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"futures returns require {sorted(required)}")
    ordered = frame.sort_values(["asset", contract_column, "date"], kind="stable")
    result = ordered.groupby(["asset", contract_column], sort=False)[price_column].pct_change(
        fill_method=None
    )
    result.index = ordered.index
    return result.sort_index()
