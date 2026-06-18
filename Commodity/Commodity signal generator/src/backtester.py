"""
Backtester vectorisé.

Conventions anti-look-ahead :
  - Les signaux en t sont calculés sur données <= t.
  - L'exécution se fait en t + execution_lag (défaut t+1).
  - Les coûts de transaction sont prélevés sur le turnover effectif.
"""

from __future__ import annotations

import pandas as pd


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame,
                 cfg: dict) -> dict:
    """Exécute le backtest et retourne PnL net, brut, coûts et turnover."""
    lag = cfg["execution_lag"]
    w = weights.shift(lag).fillna(0.0)

    gross_pnl = (w * returns).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * cfg["transaction_cost_bps"] / 1e4

    net_pnl = gross_pnl - costs

    return {
        "net": net_pnl,
        "gross": gross_pnl,
        "costs": costs,
        "turnover": turnover,
        "weights_executed": w,
        "asset_pnl": w * returns,
    }


def signal_attribution(individual_signals: dict, returns: pd.DataFrame,
                       prices_index: pd.Index, build_positions, cfg_port,
                       cfg_sig) -> pd.DataFrame:
    """Backtest chaque signal isolément (mêmes contraintes de portefeuille)
    pour produire une attribution de performance par sleeve."""
    out = {}
    for name, sig in individual_signals.items():
        w = build_positions(sig.fillna(0.0), returns, cfg_port)
        res = run_backtest(w, returns, cfg_port)
        out[name] = res["net"]
    return pd.DataFrame(out)
