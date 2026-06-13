"""Signals for the two strategies.

1. Variance Risk Premium (VRP)
   VRP_t = IV_t^2 - E_t[RV_{t->t+h}^2]. Empirically IV systematically exceeds
   subsequently-realized vol (sellers of options earn a premium for bearing
   variance/crash risk). The tradable signal is the *level* of the premium and
   its deviations; we trade a delta-hedged short-variance position when the
   premium is rich and step aside / go long when it is cheap or negative.

2. Dispersion (implied vs realized correlation)
   Index variance = sum of constituent variances + cross-covariances. Holding
   weights and constituent vols fixed, index IV implies an *average implied
   correlation*. When implied correlation sits above realized correlation, the
   dispersion trade (short index vol / long single-name vol) is rich. We
   compute both correlations and trade their spread.

Both signals are computed with strict alignment so that a position taken at t
only uses information available at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_risk_premium(
    iv: pd.Series, fwd_rv: pd.Series,
) -> pd.DataFrame:
    """Compute the variance risk premium series.

    Parameters
    ----------
    iv : pd.Series
        Implied vol (e.g. VIX/100), as a decimal, observed at t.
    fwd_rv : pd.Series
        Realized vol over the forward horizon, aligned to t.

    Returns a frame with iv, fwd_rv, vrp (in variance points) and the
    vol-point premium (iv - fwd_rv).
    """
    df = pd.DataFrame({"iv": iv, "fwd_rv": fwd_rv}).dropna()
    df["vrp_var"] = df["iv"] ** 2 - df["fwd_rv"] ** 2     # variance points
    df["vrp_vol"] = df["iv"] - df["fwd_rv"]               # vol points
    return df


def implied_correlation(
    index_iv: pd.Series, constituent_iv: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Average implied correlation backed out from index vs constituent IV.

    Using the variance decomposition with a single average correlation rho:

        sigma_I^2 = sum_i w_i^2 sigma_i^2
                    + rho * sum_{i!=j} w_i w_j sigma_i sigma_j

    Solving for rho given index IV and constituent IVs. This is the standard
    'implied correlation index' construction (cf. CBOE COR/ICJ methodology).
    """
    w = weights / weights.sum()
    out = {}
    for t in index_iv.index:
        if t not in constituent_iv.index:
            continue
        sig = constituent_iv.loc[t].dropna()
        ww = w.reindex(sig.index).dropna()
        sig = sig.reindex(ww.index)
        if len(sig) < 5:
            continue
        sig_i = float(index_iv.loc[t])
        sum_sq = float((ww**2 * sig**2).sum())
        weighted = float((ww * sig).sum())
        denom = weighted**2 - sum_sq  # = sum_{i!=j} w_i w_j sig_i sig_j
        if denom <= 0:
            continue
        rho = (sig_i**2 - sum_sq) / denom
        out[t] = np.clip(rho, -1.0, 1.0)
    return pd.Series(out, name="implied_corr")


def realized_correlation(
    returns: pd.DataFrame, window: int = 21,
) -> pd.Series:
    """Average pairwise realized correlation across the basket over a window."""
    out = {}
    cols = returns.columns
    n = len(cols)
    if n < 2:
        return pd.Series(dtype=float)
    for i in range(window, len(returns)):
        w = returns.iloc[i - window:i].dropna(axis=1, how="any")
        if w.shape[1] < 2:
            continue
        corr = w.corr().values
        # Mean of off-diagonal entries.
        off = (corr.sum() - np.trace(corr)) / (corr.shape[0] * (corr.shape[0] - 1))
        out[returns.index[i]] = off
    return pd.Series(out, name="realized_corr")


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score of a signal, lagged so it is strictly known at t.

    The rolling mean/std are computed through the *previous* observation
    (shift(1)) so the z-score at t uses only data available before t — no
    same-bar look-ahead into the value being standardized.
    """
    lagged = series.shift(1)
    mu = lagged.rolling(window).mean()
    sd = lagged.rolling(window).std()
    return (series - mu) / sd
