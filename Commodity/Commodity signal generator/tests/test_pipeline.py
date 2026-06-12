"""
Tests de non-régression du pipeline.

Le test critique est l'absence de look-ahead : tronquer l'historique ne
doit modifier ni les signaux ni les poids antérieurs à la troncature.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.data_loader import _synthetic_prices, compute_returns
from src.signals import build_composite
from src.portfolio import build_positions
from src.backtester import run_backtest


def _setup(n_days_end=None):
    prices = _synthetic_prices(list(config.UNIVERSE)[:8],
                               "2015-01-01", "2024-01-01")
    if n_days_end:
        prices = prices.iloc[:n_days_end]
    returns = compute_returns(prices)
    comp, indiv = build_composite(prices, returns, config.SIGNALS_CONFIG)
    return prices, returns, comp, indiv


def test_no_lookahead_signals():
    """Les signaux en t ne doivent pas dépendre des données > t."""
    _, _, full, _ = _setup()
    _, _, trunc, _ = _setup(n_days_end=1500)
    a = full.iloc[:1400].fillna(0)
    b = trunc.iloc[:1400].fillna(0)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-10), \
        "Look-ahead détecté dans les signaux"


def test_no_lookahead_weights():
    """Les poids en t ne doivent pas dépendre des données > t
    (hors effet de bande de non-trading, vérifié loin de la fin)."""
    prices, returns, comp, _ = _setup()
    w_full = build_positions(comp.fillna(0.0), returns,
                             config.PORTFOLIO_CONFIG)
    p2, r2, c2, _ = _setup(n_days_end=1500)
    w_trunc = build_positions(c2.fillna(0.0), r2, config.PORTFOLIO_CONFIG)
    a = w_full.iloc[:1400].to_numpy()
    b = w_trunc.iloc[:1400].to_numpy()
    assert np.allclose(a, b, atol=1e-10), "Look-ahead détecté dans les poids"


def test_execution_lag():
    """Le PnL du jour t doit utiliser les poids décidés au plus tard en t-1."""
    prices, returns, comp, _ = _setup()
    w = build_positions(comp.fillna(0.0), returns, config.PORTFOLIO_CONFIG)
    res = run_backtest(w, returns, config.PORTFOLIO_CONFIG)
    manual = (w.shift(config.PORTFOLIO_CONFIG["execution_lag"]).fillna(0.0)
              * returns).sum(axis=1)
    assert np.allclose(res["gross"].fillna(0), manual.fillna(0), atol=1e-12)


def test_vol_targeting_sane():
    """La vol réalisée doit être de l'ordre de la cible (±60%)."""
    prices, returns, comp, _ = _setup()
    w = build_positions(comp.fillna(0.0), returns, config.PORTFOLIO_CONFIG)
    res = run_backtest(w, returns, config.PORTFOLIO_CONFIG)
    pnl = res["net"].loc[res["net"].ne(0).idxmax():]
    realized = pnl.std() * np.sqrt(252)
    target = config.PORTFOLIO_CONFIG["vol_target_portfolio"]
    assert 0.4 * target < realized < 1.6 * target, \
        f"Vol réalisée {realized:.1%} hors plage vs cible {target:.0%}"


def test_signals_bounded():
    """Tous les signaux composites doivent rester dans [-1, 1]."""
    _, _, comp, indiv = _setup()
    assert comp.abs().max().max() <= 1.0 + 1e-9
    for name, s in indiv.items():
        assert s.abs().max().max() <= 1.0 + 1e-9, f"{name} hors bornes"


def test_carry_from_curve_sign_convention():
    """Backwardation (courbe décroissante) doit donner un carry > 0."""
    import datetime as dt
    from src.carry import carry_from_curve
    today = dt.date(2026, 6, 11)
    mats = [dt.date(2026, 7, 15), dt.date(2026, 10, 15),
            dt.date(2027, 1, 15), dt.date(2027, 7, 15)]
    back = pd.Series([100.0, 97.0, 94.0, 90.0], index=mats)
    cont = pd.Series([100.0, 102.0, 104.0, 107.0], index=mats)
    c1, m1 = carry_from_curve(back, today)
    c2, m2 = carry_from_curve(cont, today)
    assert c1 > 0 and c2 < 0
    assert m1 == "seasonal_pair_12m"  # juillet → juillet présent


def test_carry_history_dollar_neutral():
    """Le signal carry historique doit être dollar-neutre par date."""
    from src.carry import carry_signal_from_history
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=100)
    cols = list("ABCDE")
    f1 = pd.DataFrame(rng.uniform(50, 100, (100, 5)), idx, cols)
    f2 = f1 * rng.uniform(0.95, 1.05, (100, 5))
    sig = carry_signal_from_history(f1, f2, dt_years=1 / 12)
    assert np.allclose(sig.sum(axis=1), 0, atol=1e-9)
    assert sig.abs().max().max() <= 1 + 1e-9


if __name__ == "__main__":
    for fn in [test_no_lookahead_signals, test_no_lookahead_weights,
               test_execution_lag, test_vol_targeting_sane,
               test_signals_bounded, test_carry_from_curve_sign_convention,
               test_carry_history_dollar_neutral]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("\nTous les tests passent.")
