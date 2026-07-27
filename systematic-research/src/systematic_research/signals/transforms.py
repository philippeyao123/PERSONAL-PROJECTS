"""Past-only normalization transforms."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def _validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "feature", "available_at"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"feature frame requires {sorted(required)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    return result.sort_values(["asset", "date"], kind="stable")


def rolling_zscore(
    frame: pd.DataFrame, window: int, min_periods: Optional[int] = None
) -> pd.DataFrame:
    """Normalize with mean and volatility estimated strictly before each row."""
    if window < 2:
        raise DataValidationError("z-score window must be >= 2")
    data = _validate_feature_frame(frame)
    minimum = min_periods or window
    grouped = data.groupby("asset", sort=False)["feature"]
    past_mean = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=minimum).mean()
    )
    past_std = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=minimum).std(ddof=1)
    )
    data["score"] = (data["feature"] - past_mean) / past_std.replace(0.0, np.nan)
    return data


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Map same-date cross-sectional ranks to [-1, 1]."""
    data = _validate_feature_frame(frame)
    percentile = data.groupby("date", sort=False)["feature"].rank(pct=True, method="average")
    data["score"] = 2.0 * percentile - 1.0
    return data


def winsorize_past(
    frame: pd.DataFrame,
    window: int,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip using per-asset quantiles fitted strictly on preceding observations."""
    if not 0 <= lower < upper <= 1 or window < 2:
        raise DataValidationError("winsorization parameters are invalid")
    data = _validate_feature_frame(frame)
    grouped = data.groupby("asset", sort=False)["feature"]
    low = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).quantile(lower)
    )
    high = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).quantile(upper)
    )
    data["feature"] = data["feature"].clip(lower=low, upper=high)
    return data
