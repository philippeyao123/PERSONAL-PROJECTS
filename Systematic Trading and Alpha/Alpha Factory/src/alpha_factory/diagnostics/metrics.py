"""Diagnostics: performance, IC, and the statistical-rigor metrics that
separate a credible systematic project from a curve-fit one.

The headline differentiators:
    - Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): corrects the
      observed Sharpe for the number of trials, non-normality (skew/kurtosis),
      and sample length. A high gross Sharpe that does not survive deflation
      is, honestly, noise.
    - Probabilistic Sharpe Ratio (PSR): P(true SR > benchmark SR).
    - Capacity analysis: the AUM at which impact erodes the edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceStats:
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    skew: float
    kurtosis: float
    mean_turnover: float
    hit_rate: float

    def to_series(self) -> pd.Series:
        return pd.Series(self.__dict__)


def performance_stats(
    returns: pd.Series, turnover: pd.Series | None = None,
    periods_per_year: float = 12.0,
) -> PerformanceStats:
    """Compute standard performance statistics.

    `periods_per_year` should match the return frequency (e.g. 12 for monthly
    rebalances, 252 for daily).
    """
    r = returns.dropna()
    if len(r) < 2:
        raise ValueError("need at least 2 return observations")

    ann_return = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)
    sortino = ann_return / downside if downside > 0 else 0.0

    curve = (1 + r).cumprod()
    dd = (curve / curve.cummax() - 1.0)
    max_dd = dd.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    return PerformanceStats(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r)),  # excess kurtosis
        mean_turnover=float(turnover.mean()) if turnover is not None else np.nan,
        hit_rate=float((r > 0).mean()),
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark: float = 0.0,
    periods_per_year: float = 12.0,
) -> float:
    """PSR: probability the true (annualized) Sharpe exceeds `sr_benchmark`.

    Accounts for skew and kurtosis of the return distribution.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3:
        return np.nan
    sr = (r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0  # per-period
    sr_bench_per = sr_benchmark / np.sqrt(periods_per_year)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r) + 3.0  # convert excess -> raw kurtosis
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    if denom <= 0:
        return np.nan
    z = (sr - sr_bench_per) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, sr_variance: float | None = None,
    periods_per_year: float = 12.0,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Parameters
    ----------
    n_trials : int
        Number of independent strategy configurations tried. Higher => more
        severe deflation of the observed Sharpe.
    sr_variance : float, optional
        Variance of the Sharpe estimates across trials. If None, a conservative
        default of 1.0 (per-period) is assumed.

    Returns
    -------
    dict with the expected-maximum benchmark Sharpe under the null (E[max SR]),
    the deflated Sharpe (a probability), and the observed annualized Sharpe.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or n_trials < 1:
        return {"dsr": np.nan, "sr_annual": np.nan, "expected_max_sr": np.nan}

    sr_per = (r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    sr_annual = sr_per * np.sqrt(periods_per_year)

    if sr_variance is None:
        sr_variance = 1.0 / n  # variance of SR estimator under H0 ~ 1/n

    # Expected maximum Sharpe across N trials under the null (per-period),
    # via the expected value of the max of N standard normals.
    emc = 0.5772156649  # Euler-Mascheroni
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    expected_max = np.sqrt(sr_variance) * ((1 - emc) * z1 + emc * z2)

    g3 = stats.skew(r)
    g4 = stats.kurtosis(r) + 3.0
    denom = np.sqrt(1 - g3 * sr_per + (g4 - 1) / 4 * sr_per**2)
    if denom <= 0:
        dsr = np.nan
    else:
        z = (sr_per - expected_max) * np.sqrt(n - 1) / denom
        dsr = float(stats.norm.cdf(z))

    return {
        "dsr": dsr,
        "sr_annual": sr_annual,
        "expected_max_sr": float(expected_max * np.sqrt(periods_per_year)),
    }


def information_coefficient(ic_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-factor IC: mean, std, IR (mean/std), and t-stat."""
    if ic_frame.empty:
        return pd.DataFrame()
    out = {}
    for col in ic_frame.columns:
        ic = ic_frame[col].dropna()
        if len(ic) < 2:
            continue
        mean, std = ic.mean(), ic.std(ddof=1)
        ir = mean / std if std > 0 else np.nan
        tstat = ir * np.sqrt(len(ic)) if not np.isnan(ir) else np.nan
        out[col] = {"ic_mean": mean, "ic_std": std, "ic_ir": ir,
                    "t_stat": tstat, "n": len(ic)}
    return pd.DataFrame(out).T


def capacity_analysis(
    gross_sharpe: float, mean_turnover: float, ann_return: float,
    impact_coef_bps: float = 10.0, periods_per_year: float = 12.0,
    aum_grid: np.ndarray | None = None, adv_total: float = 1e9,
) -> pd.DataFrame:
    """Estimate how net performance decays with AUM via square-root impact.

    Crude but communicative: at higher AUM the same turnover trades a larger
    fraction of ADV, so impact (∝ sqrt(participation)) grows and eats return.
    Reports the AUM where net annual return crosses zero — the 'capacity'.
    """
    if aum_grid is None:
        aum_grid = np.logspace(7, 11, 25)  # $10M to $100B

    rows = []
    for aum in aum_grid:
        participation = (aum * mean_turnover * periods_per_year) / adv_total
        impact_bps = impact_coef_bps * np.sqrt(max(participation, 1e-12))
        annual_impact = (impact_bps / 1e4) * mean_turnover * periods_per_year
        net = ann_return - annual_impact
        rows.append({"aum": aum, "participation": participation,
                     "annual_impact": annual_impact, "net_return": net})
    df = pd.DataFrame(rows)
    return df
