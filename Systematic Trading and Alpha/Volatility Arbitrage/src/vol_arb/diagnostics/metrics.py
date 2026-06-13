"""Performance and risk diagnostics for the vol-arb strategies.

Beyond Sharpe, the metric that matters for a short-vol strategy is the
*crash profile*: selling variance earns a steady premium punctuated by large
losses (2018 Volmageddon, March 2020). A credible vol-arb writeup shows this
explicitly rather than hiding behind an average Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerfStats:
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    skew: float
    kurtosis: float
    hit_rate: float
    worst_day: float
    var_95: float
    cvar_95: float

    def to_series(self) -> pd.Series:
        return pd.Series(self.__dict__)


def performance_stats(pnl: pd.Series, periods_per_year: int = 252) -> PerfStats:
    r = pnl.dropna()
    if len(r) < 3:
        raise ValueError("need >=3 observations")
    ann_return = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    downside = r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)
    sortino = ann_return / downside if downside > 0 else 0.0
    curve = r.cumsum()
    dd = (curve - curve.cummax()).min()
    return PerfStats(
        ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
        max_drawdown=float(dd), skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r)), hit_rate=float((r > 0).mean()),
        worst_day=float(r.min()),
        var_95=float(r.quantile(0.05)),
        cvar_95=float(r[r <= r.quantile(0.05)].mean()),
    )


def vrp_summary(vrp_df: pd.DataFrame) -> dict[str, float]:
    """Average premium and how often IV exceeded subsequent RV."""
    return {
        "mean_vrp_vol_pts": float(vrp_df["vrp_vol"].mean() * 100),
        "median_vrp_vol_pts": float(vrp_df["vrp_vol"].median() * 100),
        "pct_iv_above_rv": float((vrp_df["vrp_vol"] > 0).mean() * 100),
        "mean_iv_pct": float(vrp_df["iv"].mean() * 100),
        "mean_fwd_rv_pct": float(vrp_df["fwd_rv"].mean() * 100),
    }
