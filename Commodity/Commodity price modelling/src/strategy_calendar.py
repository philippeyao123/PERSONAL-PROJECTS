"""
Calendar-spread strategy driven by the Schwartz-Smith short-term factor.

Economics: the log spread  X_t = ln F(front) - ln F(back)  loads on the
factors as  (e^{-kappa tau_f} - e^{-kappa tau_b}) chi_t  + (A terms):
the long-term factor xi cancels exactly (loading 1 on every maturity), so a
1-vs-1 log calendar spread is a *pure chi trade*. Since chi is OU, the spread
mean-reverts with the same half-life — fade it when chi is stretched.

Out-of-sample protocol (no look-ahead):
  1. Estimate Schwartz-Smith parameters on the first part of the sample only.
  2. Run the Kalman filter through the remainder with FROZEN parameters —
     filtering at t uses data up to t only, so the chi path is tradeable.
  3. Trade the front-vs-back log spread on the OOS window:
       short spread (short front / long back) when standardised chi > +z_in,
       long when < -z_in, exit inside z_out. Execution next close, 2bp costs
       per leg on turnover.

Bands come either from a heuristic z-score or from Leung-Li optimal levels
computed on chi's OU parameters (kappa, sigma_chi from the calibration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models.schwartz_smith import fit_schwartz_smith, _kalman_loglik, SSParams


@dataclass
class CalendarBacktest:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    chi: pd.Series
    spread: pd.Series
    stats: dict
    params_train: SSParams

    def summary(self) -> str:
        s = self.stats
        return (f"OOS {s['n_days']}d | CAGR={s['cagr']:+.1%} | vol={s['vol']:.1%} | "
                f"Sharpe={s['sharpe']:+.2f} | MaxDD={s['max_dd']:.1%} | "
                f"trades={s['n_trades']}")


def filter_oos(params: SSParams, log_f: pd.DataFrame, tau: pd.DataFrame,
               dt: float = 1 / 252) -> pd.DataFrame:
    """Run the Kalman filter over (log_f, tau) with frozen parameters."""
    theta = np.array([np.log(params.kappa), np.log(params.sigma_chi),
                      params.lambda_chi, params.mu_xi, params.mu_xi_star,
                      np.log(params.sigma_xi), np.arctanh(params.rho),
                      np.log(params.s_err)])
    _, states = _kalman_loglik(theta, log_f.values, tau.values, dt,
                               return_states=True)
    return pd.DataFrame(states, index=log_f.index, columns=["chi", "xi"])


def run_calendar_strategy(log_f: pd.DataFrame, tau: pd.DataFrame,
                          train_frac: float = 0.5,
                          entry: float = 1.0, exit_: float = 0.25,
                          band_mode: str = "revert",
                          bands_in_chi_units: bool = False,
                          cost_bps: float = 2.0,
                          front: str | None = None, back: str | None = None
                          ) -> CalendarBacktest:
    """
    band_mode:
      "revert" — z-score style: enter outside ±entry, exit once |signal| < exit_.
      "cross"  — Leung-Li style: a long entered at signal <= -entry is held
                 until signal >= +exit_ on the *other side* of the mean
                 (and symmetrically for shorts). With Leung-Li thresholds
                 pass entry=|d*|, exit_=|b*| and bands_in_chi_units=True.
    If bands_in_chi_units: `entry`/`exit_` are absolute chi levels;
    otherwise z-scores of chi standardised by its OU stationary std
    from the training fit.
    """
    n_train = int(len(log_f) * train_frac)
    params, _ = fit_schwartz_smith(log_f.iloc[:n_train], tau.iloc[:n_train])
    factors = filter_oos(params, log_f, tau)        # filtered, causal
    chi = factors["chi"].iloc[n_train:]

    # front = shortest average maturity, back = longest
    if front is None:
        front = tau.mean().idxmin()
    if back is None:
        back = tau.mean().idxmax()
    spread = (log_f[front] - log_f[back]).iloc[n_train:]

    sd_stat = params.sigma_chi / np.sqrt(2 * params.kappa)  # chi stationary std
    sig = chi if bands_in_chi_units else chi / sd_stat

    pos = pd.Series(0.0, index=sig.index)
    state = 0.0
    for t, s in sig.items():
        if band_mode == "revert":
            if state == 0.0:
                state = -1.0 if s > entry else (1.0 if s < -entry else 0.0)
            elif abs(s) < exit_:
                state = 0.0
        else:  # "cross": hold through the mean to the opposite threshold
            if state == 0.0:
                state = -1.0 if s > entry else (1.0 if s < -entry else 0.0)
            elif state == 1.0 and s >= exit_:
                state = 0.0
            elif state == -1.0 and s <= -exit_:
                state = 0.0
        pos[t] = state

    pos_lag = pos.shift(1).fillna(0.0)
    r_f = log_f[front].diff().iloc[n_train:]
    r_b = log_f[back].diff().iloc[n_train:]
    gross = pos_lag * (r_f - r_b)
    turnover = pos_lag.diff().abs().fillna(0.0) * 2          # two legs
    net = (gross - turnover * cost_bps / 1e4).dropna()

    equity = (1 + net).cumprod()
    vol = net.std() * np.sqrt(252)
    sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan
    stats = {
        "n_days": len(net),
        "cagr": equity.iloc[-1] ** (252 / len(net)) - 1,
        "vol": vol, "sharpe": sharpe,
        "max_dd": (equity / equity.cummax() - 1).min(),
        "n_trades": int((pos_lag.diff().abs() > 0).sum()),
    }
    return CalendarBacktest(equity, net, pos_lag, chi, spread, stats, params)
