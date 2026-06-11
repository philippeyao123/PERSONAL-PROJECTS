"""
Stat-arb on the WTI-Brent spread, signal driven by a rolling OU fit.

Spread construction:
    spread_t = log(WTI_t) - beta_t * log(Brent_t),
    beta_t from a rolling OLS (hedge ratio re-estimated daily, no look-ahead).

Signal:
    z_t = (spread_t - rolling mean) / rolling std
    enter short spread at z > +z_in, long at z < -z_in, exit at |z| < z_out.
    Positions lagged one day before P&L is computed (trade at next close).

Costs: proportional, applied on turnover of both legs.
This mirrors the classic Engle-Granger style pairs framework but with the
hedge ratio and bands fully out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    zscore: pd.Series
    spread: pd.Series
    stats: dict

    def summary(self) -> str:
        s = self.stats
        return (f"CAGR={s['cagr']:.1%} | vol={s['vol']:.1%} | Sharpe={s['sharpe']:.2f} | "
                f"MaxDD={s['max_dd']:.1%} | hit={s['hit_rate']:.0%} | "
                f"trades={s['n_trades']} | VaR95(1d)={s['var_95']:.2%} | ES95={s['es_95']:.2%}")


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def backtest_spread(px: pd.DataFrame, leg_y: str = "WTI", leg_x: str = "Brent",
                    beta_window: int = 120, z_window: int = 60,
                    z_in: float = 2.0, z_out: float = 0.5,
                    cost_bps: float = 2.0) -> BacktestResult:
    ly, lx = np.log(px[leg_y]), np.log(px[leg_x])
    beta = _rolling_beta(ly, lx, beta_window).shift(1)      # known at t-1
    spread = ly - beta * lx
    mu = spread.rolling(z_window).mean()
    sd = spread.rolling(z_window).std()
    z = (spread - mu) / sd

    # state machine: -1 short spread, +1 long spread, 0 flat
    pos = pd.Series(0.0, index=z.index)
    state = 0.0
    for t, zt in z.items():
        if np.isnan(zt):
            pos[t] = 0.0
            continue
        if state == 0.0:
            state = -1.0 if zt > z_in else (1.0 if zt < -z_in else 0.0)
        elif abs(zt) < z_out:
            state = 0.0
        pos[t] = state

    pos_lag = pos.shift(1).fillna(0.0)                      # trade next close
    r_y, r_x = ly.diff(), lx.diff()
    gross = pos_lag * (r_y - beta * r_x)
    turnover = (pos_lag.diff().abs() * (1 + beta.abs())).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    net = net.dropna()

    equity = (1 + net).cumprod()
    ann = 252
    cagr = equity.iloc[-1] ** (ann / len(net)) - 1
    vol = net.std() * np.sqrt(ann)
    sharpe = net.mean() / net.std() * np.sqrt(ann) if net.std() > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    trades = int((pos_lag.diff().abs() > 0).sum())
    active = net[pos_lag.reindex(net.index) != 0]
    hit = (active > 0).mean() if len(active) else np.nan
    var95 = -np.percentile(net, 5)
    es95 = -net[net <= -var95].mean() if (net <= -var95).any() else np.nan

    return BacktestResult(
        equity=equity, returns=net, positions=pos_lag, zscore=z, spread=spread,
        stats={"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd,
               "hit_rate": hit, "n_trades": trades, "var_95": var95, "es_95": es95},
    )
