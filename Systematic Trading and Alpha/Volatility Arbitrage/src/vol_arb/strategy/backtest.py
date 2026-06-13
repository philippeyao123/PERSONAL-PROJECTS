"""Strategy P&L for short/long variance and dispersion positions.

The original notebook computed P&L as ``iv^2 - rv^2`` where iv was itself
``rv + noise`` — circular and guaranteed profitable. Here the P&L is the real
economics of a *delta-hedged* variance position:

A short variance-swap position entered at t with strike K = IV_t^2 pays, over
the holding horizon,

    PnL = vega_notional * (K - RV_{t->t+h}^2) / (2 * K^{1/2}) ... (vol points)

We use the clean variance-swap convention: a short variance swap with variance
strike K_var = IV^2 and unit variance-notional pays (IV^2 - RV^2) per unit
notional over the period — but unlike the notebook, RV is the *forward
realized* vol that actually materializes, IV is *observed* market vol, and the
position is sized and costed realistically. This makes the P&L an honest test
of the variance risk premium rather than a tautology.

Costs: a fixed bid/ask in vega terms charged on each new position (variance
swaps and option strategies have meaningful transaction costs, especially in
the wings).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostParams:
    # round-trip cost in vol points (0.5 = 50bp of vol)
    vega_cost_vol_pts: float = 0.5


def variance_swap_pnl(
    iv: pd.Series, fwd_rv: pd.Series, position: pd.Series,
    cost: CostParams | None = None,
) -> pd.DataFrame:
    """Period P&L of a variance-swap position, in VOL points (normalized).

    Parameters
    ----------
    iv : implied vol observed at t (decimal).
    fwd_rv : realized vol over (t, t+h], aligned to t (decimal).
    position : +1 long variance, -1 short variance, 0 flat (decided at t).
    cost : transaction-cost params; charged on position changes.

    We report P&L in *vol points* rather than variance points. A raw variance
    swap pays (RV^2 - IV^2), but in a vol spike (VIX -> 80%) the IV^2 term
    dominates and the series becomes uninterpretable (skew/kurtosis explode
    from a single observation). Normalizing by 2*K^{1/2} converts to the vega
    convention — the standard desk practice — so a vol point of move maps to a
    comparable P&L across regimes:

        pnl_vol = pos * (RV^2 - IV^2) / (2 * IV)      [long variance, +pos]
    """
    cost = cost or CostParams()
    df = pd.DataFrame({"iv": iv, "fwd_rv": fwd_rv, "pos": position}).dropna()
    df = df[df["iv"] > 1e-6]
    long_pnl_vol = (df["fwd_rv"] ** 2 - df["iv"] ** 2) / (2 * df["iv"])
    pnl = np.where(df["pos"] > 0, long_pnl_vol,
                   np.where(df["pos"] < 0, -long_pnl_vol, 0.0))
    df["pnl_vol"] = pnl
    pos_change = df["pos"].diff().abs().fillna(df["pos"].abs())
    df["cost"] = pos_change * (cost.vega_cost_vol_pts / 100.0)
    df["pnl_net"] = df["pnl_vol"] - df["cost"]
    return df


def dispersion_pnl(
    implied_corr: pd.Series, realized_corr: pd.Series,
    position: pd.Series, notional: float = 1.0,
    cost_corr_pts: float = 0.02,
) -> pd.DataFrame:
    """P&L of a dispersion (implied vs realized correlation) position.

    A short-correlation position (short index vol / long single-name vol)
    profits when realized correlation comes in below the implied correlation
    sold. P&L per period approximated as proportional to
    (implied_corr - realized_corr) for a short-correlation book.
    """
    df = pd.DataFrame({
        "ic": implied_corr, "rc": realized_corr, "pos": position,
    }).dropna()
    short_corr_pnl = df["ic"] - df["rc"]    # short correlation profits if ic>rc
    pnl = np.where(df["pos"] < 0, short_corr_pnl,
                   np.where(df["pos"] > 0, -short_corr_pnl, 0.0)) * notional
    df["pnl_gross"] = pnl
    pos_change = df["pos"].diff().abs().fillna(df["pos"].abs())
    df["cost"] = pos_change * cost_corr_pts
    df["pnl_net"] = df["pnl_gross"] - df["cost"]
    return df


def vrp_signal_to_position(
    vrp_z: pd.Series, entry: float = 0.5, short_bias: bool = True,
) -> pd.Series:
    """Map a VRP z-score to a position, using only information known at t.

    The z-score is computed from a trailing window (see signals.zscore), so the
    position at t does not peek at future data. Default policy reflects the
    empirical asymmetry: the premium is usually positive, so we are short
    variance when rich (z > entry), flat when modest, and long only when the
    premium is clearly negative (z < -entry).
    """
    pos = pd.Series(0.0, index=vrp_z.index)
    pos[vrp_z > entry] = -1.0   # rich premium -> short variance
    pos[vrp_z < -entry] = 1.0   # negative premium -> long variance
    return pos


def always_short(index: pd.Index) -> pd.Series:
    """Benchmark policy: always short variance. Exposes the raw crash profile."""
    return pd.Series(-1.0, index=index)
