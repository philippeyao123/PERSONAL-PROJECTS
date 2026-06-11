"""
Métriques de performance et de risque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def perf_summary(pnl: pd.Series, turnover: pd.Series | None = None,
                 rf: float = 0.0) -> dict:
    """Tableau de bord complet des métriques annualisées."""
    pnl = pnl.dropna()
    mu = pnl.mean() * ANN
    sigma = pnl.std() * np.sqrt(ANN)
    downside = pnl[pnl < 0].std() * np.sqrt(ANN)

    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = dd.min()

    # Durée du drawdown max
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()

    sharpe = (mu - rf) / sigma if sigma > 0 else np.nan
    n = len(pnl)
    sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / (n / ANN)) if n else np.nan

    out = {
        "Ann. Return": mu,
        "Ann. Volatility": sigma,
        "Sharpe Ratio": sharpe,
        "Sharpe SE (Lo, 2002)": sharpe_se,
        "Sortino Ratio": (mu - rf) / downside if downside > 0 else np.nan,
        "Max Drawdown": max_dd,
        "Calmar Ratio": mu / abs(max_dd) if max_dd < 0 else np.nan,
        "Skewness": stats.skew(pnl),
        "Excess Kurtosis": stats.kurtosis(pnl),
        "Hit Rate": (pnl > 0).mean(),
        "VaR 95% (daily)": np.percentile(pnl, 5),
        "ES 95% (daily)": pnl[pnl <= np.percentile(pnl, 5)].mean(),
        "Worst Day": pnl.min(),
        "Best Day": pnl.max(),
        "DD Peak": str(start.date()),
        "DD Trough": str(end.date()),
        "Obs (years)": n / ANN,
    }
    if turnover is not None:
        out["Ann. Turnover (x gross)"] = turnover.mean() * ANN
    return out


def rolling_sharpe(pnl: pd.Series, window: int = 252) -> pd.Series:
    mu = pnl.rolling(window).mean() * ANN
    sd = pnl.rolling(window).std() * np.sqrt(ANN)
    return mu / sd


def drawdown_series(pnl: pd.Series) -> pd.Series:
    equity = (1 + pnl.fillna(0)).cumprod()
    return equity / equity.cummax() - 1


def print_summary(summary: dict, title: str = "PERFORMANCE SUMMARY") -> None:
    print(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")
    for k, v in summary.items():
        if isinstance(v, str):
            print(f"{k:<28}: {v}")
        elif "Rate" in k or "Return" in k or "Vol" in k or "Drawdown" in k \
                or "VaR" in k or "ES" in k or "Day" in k:
            print(f"{k:<28}: {v:>10.2%}")
        else:
            print(f"{k:<28}: {v:>10.2f}")
    print("=" * 60)
