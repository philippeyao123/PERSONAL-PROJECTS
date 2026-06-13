"""Factor preprocessing and combination.

The transformation pipeline applied at each rebalance date:

    raw -> winsorize -> sector/size neutralize -> z-score -> combine

Combination methods:
    - "equal"   : simple average of standardized factors
    - "ic"      : IC-weighted average (weights from rolling factor IC)
    - "ridge"   : light ridge regression of forward returns on factors

We deliberately keep ML light here. Over-engineering the combiner (deep nets,
heavy boosting) is a classic way to overfit cross-sectional signals; reviewers
on systematic desks read it as a red flag, not sophistication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, limits: float = 0.025) -> pd.Series:
    """Clip a series at the given lower/upper quantiles."""
    if s.empty:
        return s
    lo, hi = s.quantile(limits), s.quantile(1 - limits)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    """Standardize to mean 0, std 1. Returns zeros if degenerate."""
    if s.empty or s.std(ddof=0) == 0 or np.isnan(s.std(ddof=0)):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def neutralize(scores: pd.Series, groups: pd.Series) -> pd.Series:
    """Demean scores within each group (e.g. sector neutralization).

    Removes group-level tilts so the signal is a within-group bet rather than
    an implicit sector bet.
    """
    aligned = groups.reindex(scores.index)
    return scores - scores.groupby(aligned).transform("mean")


class FactorCombiner:
    """Standardizes and combines multiple factor score Series into one signal.

    Parameters
    ----------
    method : {"equal", "ic", "ridge"}
    winsor_limits : float
        Tail fraction clipped on each side before standardization.
    ridge_lambda : float
        L2 penalty for the "ridge" method.
    """

    def __init__(
        self,
        method: str = "equal",
        winsor_limits: float = 0.025,
        ridge_lambda: float = 1.0,
    ) -> None:
        if method not in {"equal", "ic", "ridge"}:
            raise ValueError(f"unknown method '{method}'")
        self.method = method
        self.winsor_limits = winsor_limits
        self.ridge_lambda = ridge_lambda

    def _standardize(
        self, factor_scores: dict[str, pd.Series], sectors: pd.Series | None
    ) -> pd.DataFrame:
        cols = {}
        for name, s in factor_scores.items():
            s = winsorize(s, self.winsor_limits)
            if sectors is not None:
                s = neutralize(s, sectors)
            cols[name] = zscore(s)
        df = pd.DataFrame(cols)
        return df.dropna(how="all")

    def combine(
        self,
        factor_scores: dict[str, pd.Series],
        sectors: pd.Series | None = None,
        ic_weights: dict[str, float] | None = None,
        forward_returns: pd.Series | None = None,
    ) -> pd.Series:
        """Combine standardized factors into a single cross-sectional signal."""
        std = self._standardize(factor_scores, sectors)
        if std.empty:
            return pd.Series(dtype=float)
        # Common universe across all factors.
        std = std.dropna()
        if std.empty:
            return pd.Series(dtype=float)

        if self.method == "equal":
            signal = std.mean(axis=1)

        elif self.method == "ic":
            if not ic_weights:
                signal = std.mean(axis=1)
            else:
                w = pd.Series(ic_weights).reindex(std.columns).fillna(0.0)
                if w.abs().sum() == 0:
                    signal = std.mean(axis=1)
                else:
                    w = w / w.abs().sum()
                    signal = std.mul(w, axis=1).sum(axis=1)

        else:  # ridge
            if forward_returns is None:
                signal = std.mean(axis=1)
            else:
                y = forward_returns.reindex(std.index).dropna()
                x = std.reindex(y.index)
                if len(y) < std.shape[1] + 2:
                    signal = std.mean(axis=1)
                else:
                    X = x.values
                    lam = self.ridge_lambda
                    beta = np.linalg.solve(
                        X.T @ X + lam * np.eye(X.shape[1]), X.T @ y.values
                    )
                    signal = pd.Series(std.values @ beta, index=std.index)

        return zscore(signal)
