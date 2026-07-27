"""Portfolio construction, volatility scaling and hard constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.exceptions import ConstraintError, DataValidationError


def equal_weight_from_scores(
    scores: pd.DataFrame,
    *,
    long_quantile: float = 0.8,
    short_quantile: float = 0.2,
    gross_target: float = 1.0,
) -> pd.DataFrame:
    """Equal weight the upper and lower cross-sectional score tails."""
    if not 0 <= short_quantile < long_quantile <= 1 or gross_target <= 0:
        raise DataValidationError("equal-weight portfolio parameters are invalid")
    result = scores.copy()

    def weights(group: pd.DataFrame) -> pd.DataFrame:
        low = group["score"].quantile(short_quantile)
        high = group["score"].quantile(long_quantile)
        long_mask = group["score"] >= high
        short_mask = group["score"] <= low
        output = pd.Series(0.0, index=group.index)
        if long_mask.any():
            output.loc[long_mask] = 0.5 * gross_target / long_mask.sum()
        if short_mask.any():
            output.loc[short_mask] = -0.5 * gross_target / short_mask.sum()
        group = group.copy()
        group["target_weight"] = output
        return group

    groups = [weights(group) for _, group in result.groupby("date", sort=False)]
    return pd.concat(groups, ignore_index=True)


def volatility_scale(
    weights: pd.DataFrame,
    strategy_returns: pd.Series,
    *,
    target_volatility: float,
    window: int,
    periods_per_year: int = 252,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Scale using volatility estimated strictly before the allocation date."""
    if target_volatility <= 0 or window < 2 or periods_per_year <= 0 or max_leverage <= 0:
        raise DataValidationError("volatility scaling parameters are invalid")
    past_volatility = strategy_returns.shift(1).rolling(window, min_periods=window).std(
        ddof=1
    ) * np.sqrt(periods_per_year)
    scaler = (target_volatility / past_volatility).clip(upper=max_leverage).fillna(0.0)
    result = weights.copy()
    result["target_weight"] *= result["date"].map(scaler).fillna(0.0)
    return result


def enforce_constraints(
    weights: pd.DataFrame,
    *,
    gross_limit: float,
    net_limit: float,
    concentration_limit: float,
) -> pd.DataFrame:
    """Clip concentration, neutralize net exposure and rescale gross exposure."""
    if gross_limit <= 0 or net_limit < 0 or concentration_limit <= 0:
        raise ConstraintError("constraint limits are invalid")
    result = weights.copy()
    result["target_weight"] = result["target_weight"].clip(
        -concentration_limit, concentration_limit
    )
    for _, index in result.groupby("date").groups.items():
        current = result.loc[index, "target_weight"]
        if abs(current.sum()) > net_limit:
            current = current - current.mean()
        gross = current.abs().sum()
        if gross > gross_limit:
            current *= gross_limit / gross
        result.loc[index, "target_weight"] = current
    gross = result.groupby("date")["target_weight"].apply(lambda values: values.abs().sum())
    net = result.groupby("date")["target_weight"].sum().abs()
    concentration = result["target_weight"].abs().max()
    if (gross > gross_limit + 1e-12).any() or (net > net_limit + 1e-12).any():
        raise ConstraintError("portfolio remains outside gross/net limits")
    if concentration > concentration_limit + 1e-12:
        raise ConstraintError("portfolio remains outside concentration limit")
    return result
