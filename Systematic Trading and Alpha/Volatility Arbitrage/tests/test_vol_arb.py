"""Tests. The headline properties: no look-ahead in the signal, correct P&L
sign (short vol loses in a vol spike), and the empirical VRP being positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.data.loader import forward_realized_vol, realized_vol
from vol_arb.signals.vrp import (
    implied_correlation,
    realized_correlation,
    variance_risk_premium,
    zscore,
)
from vol_arb.strategy.backtest import (
    CostParams,
    variance_swap_pnl,
)


def _crash_prices():
    rets = np.concatenate([np.full(40, 0.0005), np.full(10, -0.05),
                           np.full(40, 0.0005)])
    return pd.Series(100 * np.exp(np.cumsum(rets)),
                     index=pd.bdate_range("2020-01-01", periods=90))


# ----------------------------- alignment -----------------------------
def test_forward_rv_sees_future_crash():
    p = _crash_prices()
    trail = realized_vol(p, 10)
    fwd = forward_realized_vol(p, 10)
    # 5 days before the crash, forward RV should exceed trailing RV.
    assert fwd.iloc[35] > trail.iloc[35]
    # well after, trailing exceeds forward.
    assert trail.iloc[55] > fwd.iloc[55]


def test_zscore_no_same_bar_lookahead():
    s = pd.Series(np.arange(300.0))
    z = zscore(s, window=50)
    # The implementation: mu/sd from s.shift(1).rolling(50) at t covers the
    # 50 lagged values ending at the lagged value for t (i.e. s[t-50:t]).
    t = 200
    lagged = s.shift(1)
    win = lagged.iloc[t - 49:t + 1]  # rolling(50) window ending at position t
    mu, sd = win.mean(), win.std()
    assert abs(z.iloc[t] - (s.iloc[t] - mu) / sd) < 1e-6
    # And crucially the window's max is s[t-1], never s[t] (no same-bar peek).
    assert win.max() == s.iloc[t - 1]


# ----------------------------- P&L sign -----------------------------
def test_short_variance_loses_in_vol_spike():
    # IV low at t, but forward RV high (spike) -> short variance loses.
    iv = pd.Series([0.15, 0.15, 0.15])
    fwd_rv = pd.Series([0.60, 0.60, 0.60])
    pos = pd.Series([-1.0, -1.0, -1.0])
    bt = variance_swap_pnl(iv, fwd_rv, pos, CostParams(vega_cost_vol_pts=0.0))
    assert (bt["pnl_vol"] < 0).all()


def test_short_variance_gains_when_calm():
    iv = pd.Series([0.20, 0.20, 0.20])
    fwd_rv = pd.Series([0.10, 0.10, 0.10])
    pos = pd.Series([-1.0, -1.0, -1.0])
    bt = variance_swap_pnl(iv, fwd_rv, pos, CostParams(vega_cost_vol_pts=0.0))
    assert (bt["pnl_vol"] > 0).all()


def test_long_variance_opposite_sign():
    iv = pd.Series([0.20])
    fwd_rv = pd.Series([0.10])
    short = variance_swap_pnl(iv, fwd_rv, pd.Series([-1.0]),
                              CostParams(vega_cost_vol_pts=0.0))
    long_ = variance_swap_pnl(iv, fwd_rv, pd.Series([1.0]),
                              CostParams(vega_cost_vol_pts=0.0))
    assert np.isclose(short["pnl_vol"].iloc[0], -long_["pnl_vol"].iloc[0])


def test_costs_reduce_pnl():
    iv = pd.Series([0.20, 0.20])
    fwd_rv = pd.Series([0.10, 0.10])
    pos = pd.Series([-1.0, 1.0])  # a flip -> incurs cost
    free = variance_swap_pnl(iv, fwd_rv, pos, CostParams(vega_cost_vol_pts=0.0))
    costly = variance_swap_pnl(iv, fwd_rv, pos, CostParams(vega_cost_vol_pts=2.0))
    assert costly["pnl_net"].sum() < free["pnl_net"].sum()


# ----------------------------- correlation -----------------------------
def test_implied_correlation_in_range():
    dates = pd.bdate_range("2020-01-01", periods=10)
    index_iv = pd.Series(0.20, index=dates)
    cons_iv = pd.DataFrame(0.30, index=dates, columns=["A", "B", "C"])
    w = pd.Series([1, 1, 1], index=["A", "B", "C"])
    ic = implied_correlation(index_iv, cons_iv, w)
    assert ((ic >= -1) & (ic <= 1)).all()


def test_realized_correlation_perfect_when_identical():
    dates = pd.bdate_range("2020-01-01", periods=60)
    base = np.random.default_rng(0).normal(0, 0.01, 60)
    rets = pd.DataFrame({"A": base, "B": base, "C": base}, index=dates)
    rc = realized_correlation(rets, window=21).dropna()
    assert (rc > 0.99).all()


# ----------------------------- VRP empirical -----------------------------
def test_vrp_positive_when_iv_above_rv():
    iv = pd.Series([0.20, 0.22, 0.18])
    fwd_rv = pd.Series([0.15, 0.16, 0.14])
    vrp = variance_risk_premium(iv, fwd_rv)
    assert (vrp["vrp_vol"] > 0).all()


# ----------------------------- plots -----------------------------
def test_plots_generate(tmp_path):
    from vol_arb.plots import (
        plot_correlation,
        plot_equity_curves,
        plot_iv_vs_rv,
        plot_return_distribution,
    )
    dates = pd.bdate_range("2020-01-01", periods=100)
    iv = pd.Series(0.20, index=dates)
    rv = pd.Series(0.15, index=dates)
    assert plot_iv_vs_rv(iv, rv, tmp_path / "a.png").exists()
    eq = pd.Series(np.linspace(0, 1, 100), index=dates)
    assert plot_equity_curves(eq, eq * 2, tmp_path / "b.png").exists()
    pnl = pd.Series(np.random.default_rng(0).normal(0, 0.05, 100), index=dates)
    assert plot_return_distribution(pnl, tmp_path / "c.png").exists()
    assert plot_correlation(iv, rv, tmp_path / "d.png").exists()
