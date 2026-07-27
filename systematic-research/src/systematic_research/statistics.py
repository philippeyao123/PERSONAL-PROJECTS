"""Multiple-testing diagnostics and cross-sectional information coefficients."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Tuple

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey and López de Prado PSR using non-excess kurtosis."""
    if observations < 2:
        raise DataValidationError("PSR requires at least two observations")
    denominator_squared = (
        1.0
        - skewness * observed_sharpe
        + 0.25 * (kurtosis - 1.0) * observed_sharpe * observed_sharpe
    )
    if denominator_squared <= 0:
        raise DataValidationError("PSR denominator is not positive")
    statistic = (
        (observed_sharpe - benchmark_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(denominator_squared)
    )
    return NormalDist().cdf(statistic)


def expected_maximum_sharpe(
    independent_trials: int,
    sharpe_standard_deviation: float,
) -> float:
    """Expected maximum under multiple independent normal trials."""
    if independent_trials < 1 or sharpe_standard_deviation < 0:
        raise DataValidationError("trial count and Sharpe dispersion are invalid")
    if independent_trials == 1 or sharpe_standard_deviation == 0:
        return 0.0
    euler_gamma = 0.5772156649015329
    normal = NormalDist()
    first = normal.inv_cdf(1.0 - 1.0 / independent_trials)
    second = normal.inv_cdf(1.0 - 1.0 / (independent_trials * math.e))
    return sharpe_standard_deviation * ((1.0 - euler_gamma) * first + euler_gamma * second)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_trials: pd.Series,
    observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    clean = sharpe_trials.dropna()
    if clean.empty:
        raise DataValidationError("DSR requires Sharpe estimates from attempted trials")
    threshold = expected_maximum_sharpe(len(clean), float(clean.std(ddof=0)))
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        threshold,
        observations,
        skewness,
        kurtosis,
    )


def information_coefficient(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
    forward_return_column: str = "forward_return",
    method: str = "spearman",
) -> pd.Series:
    required = {"date", score_column, forward_return_column}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"IC requires {sorted(required)}")

    def correlation(group: pd.DataFrame) -> float:
        valid = group[[score_column, forward_return_column]].dropna()
        return float(valid.corr(method=method).iloc[0, 1]) if len(valid) >= 3 else np.nan

    return frame.groupby("date", sort=True)[[score_column, forward_return_column]].apply(
        correlation
    )


def information_ratio(ic: pd.Series) -> float:
    clean = ic.dropna()
    standard_deviation = clean.std(ddof=1)
    return float(clean.mean() / standard_deviation) if standard_deviation > 0 else 0.0


def ic_decay(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    maximum_horizon: int,
) -> pd.DataFrame:
    """Cross-sectional IC by future return horizon."""
    if maximum_horizon < 1:
        raise DataValidationError("IC decay horizon must be positive")
    merged = scores[["date", "asset", "score"]].merge(
        returns[["date", "asset", "return"]], on=["date", "asset"], how="inner"
    )
    rows = []
    for horizon in range(1, maximum_horizon + 1):
        merged["forward_return"] = merged.groupby("asset")["return"].shift(-horizon)
        ic = information_coefficient(merged)
        rows.append(
            {
                "horizon": horizon,
                "mean_ic": float(ic.mean()),
                "information_ratio": information_ratio(ic),
                "observations": int(ic.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def return_moments(returns: pd.Series) -> Tuple[float, float]:
    clean = returns.dropna()
    return float(clean.skew()), float(clean.kurt() + 3.0)
