"""Portfolio construction.

Maps a cross-sectional signal into target weights. Two schemes:

    - quantile long/short : long top decile, short bottom decile, dollar-neutral
    - rank-proportional   : weights proportional to demeaned signal rank

Both produce a dollar-neutral book (sum of weights ~ 0) scaled to unit gross
exposure (sum |w| = 1), so leverage is explicit and comparable across dates.
"""
from __future__ import annotations

import pandas as pd


def quantile_long_short(
    signal: pd.Series, quantile: float = 0.1, gross: float = 1.0
) -> pd.Series:
    """Long the top `quantile`, short the bottom `quantile`, dollar-neutral.

    Within each leg, weights are equal. The book is scaled so that total gross
    exposure equals `gross`.
    """
    s = signal.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    n = max(1, int(len(s) * quantile))
    ranked = s.sort_values()
    shorts = ranked.index[:n]
    longs = ranked.index[-n:]

    w = pd.Series(0.0, index=s.index)
    w[longs] = 0.5 * gross / n
    w[shorts] = -0.5 * gross / n
    return w[w != 0.0]


def rank_proportional(signal: pd.Series, gross: float = 1.0) -> pd.Series:
    """Weights proportional to demeaned cross-sectional rank, dollar-neutral."""
    s = signal.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    ranks = s.rank()
    centered = ranks - ranks.mean()
    if centered.abs().sum() == 0:
        return pd.Series(dtype=float)
    w = centered / centered.abs().sum() * gross
    return w


def apply_position_limits(
    w: pd.Series, max_weight: float = 0.05, max_iter: int = 10
) -> pd.Series:
    """Cap absolute position size while preserving gross exposure where feasible.

    Capping shrinks gross, so we redistribute the freed budget to uncapped
    names and re-cap, iterating to a fixed point. If the cap is infeasible
    (gross / max_weight > n_names), the final book simply has every name at
    the cap and gross is necessarily reduced.
    """
    if w.empty:
        return w
    gross_target = w.abs().sum()
    capped = w.clip(-max_weight, max_weight)
    for _ in range(max_iter):
        gross_now = capped.abs().sum()
        if gross_now == 0 or abs(gross_now - gross_target) < 1e-12:
            break
        at_cap = capped.abs() >= max_weight - 1e-12
        free = ~at_cap
        if not free.any():
            break  # everything capped: infeasible to hit gross_target
        deficit = gross_target - gross_now
        free_gross = capped[free].abs().sum()
        if free_gross == 0:
            break
        scale = 1.0 + deficit / free_gross
        capped[free] = capped[free] * scale
        capped = capped.clip(-max_weight, max_weight)
    return capped
