"""
Commodity Trading Signal Generator — pipeline principal.

Usage :
    python main.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loader import load_prices, compute_returns
from src.signals import build_composite
from src.portfolio import build_positions
from src.backtester import run_backtest, signal_attribution
from src.metrics import perf_summary, print_summary
from src.plotting import tearsheet, current_signals_chart, carry_charts
from src.carry import live_carry_signal

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("[1/5] Chargement des données...")
    prices = load_prices(list(config.UNIVERSE), config.START_DATE,
                         config.END_DATE)
    returns = compute_returns(prices)
    print(f"      {prices.shape[1]} actifs, {prices.shape[0]} jours "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # --------------------------------------------------------------- signals
    print("[2/5] Génération des signaux...")
    composite, individual = build_composite(prices, returns,
                                            config.SIGNALS_CONFIG)

    # ------------------------------------------------------------- portfolio
    print("[3/5] Construction du portefeuille (vol targeting)...")
    weights = build_positions(composite.fillna(0.0), returns,
                              config.PORTFOLIO_CONFIG)

    # -------------------------------------------------------------- backtest
    print("[4/5] Backtest...")
    results = run_backtest(weights, returns, config.PORTFOLIO_CONFIG)
    summary = perf_summary(results["net"], results["turnover"])
    print_summary(summary, "COMPOSITE STRATEGY — NET OF COSTS")

    # Décomposition par sous-périodes (lecture des régimes)
    print("\nSharpe par sous-période (composite, net) :")
    for a, b in [("2011", "2015"), ("2015", "2020"), ("2020", None)]:
        sub = results["net"].loc[a:b]
        if len(sub) > 252:
            sr = sub.mean() / sub.std() * (252 ** 0.5)
            print(f"  {a} → {b or 'présent'} : Sharpe = {sr:>5.2f}")

    attribution = signal_attribution(individual, returns, prices.index,
                                     build_positions,
                                     config.PORTFOLIO_CONFIG,
                                     config.SIGNALS_CONFIG)
    print("\nSharpe par signal (standalone, net de coûts) :")
    for col in attribution.columns:
        s = perf_summary(attribution[col])
        print(f"  {col:<16}: Sharpe = {s['Sharpe Ratio']:>5.2f} | "
              f"AnnRet = {s['Ann. Return']:>7.2%} | "
              f"MaxDD = {s['Max Drawdown']:>7.2%}")

    # --------------------------------------------------------------- outputs
    print("\n[5/5] Génération des livrables...")
    tearsheet(results, attribution, individual, config.SECTORS,
              os.path.join(OUTPUT_DIR, "tearsheet.png"))
    current_signals_chart(composite, config.UNIVERSE,
                          os.path.join(OUTPUT_DIR, "current_signals.png"))

    # Export CSV : signaux courants + métriques
    last_signals = pd.DataFrame({
        "name": [config.UNIVERSE.get(t, t) for t in composite.columns],
        "sector": [config.SECTORS.get(t, "Other") for t in composite.columns],
        "composite_signal": composite.iloc[-1].round(3),
        "target_weight": weights.iloc[-1].round(3),
        **{f"sig_{k}": v.iloc[-1].round(3) for k, v in individual.items()},
    })
    last_signals.to_csv(os.path.join(OUTPUT_DIR, "current_signals.csv"))
    pd.Series(summary).to_csv(os.path.join(OUTPUT_DIR, "perf_summary.csv"))
    results["net"].to_csv(os.path.join(OUTPUT_DIR, "daily_pnl.csv"))

    # ------------------------------------------------------ carry (live)
    print("[6/6] Facteur carry — courbes à terme live...")
    try:
        carry_df, curves = live_carry_signal(config.UNIVERSE)
        carry_df.to_csv(os.path.join(OUTPUT_DIR, "current_carry.csv"))
        carry_charts(carry_df, curves, config.UNIVERSE,
                     os.path.join(OUTPUT_DIR, "carry_snapshot.png"))
        nb = (carry_df["state"] == "backwardation").sum()
        print(f"      {len(carry_df)} courbes construites — "
              f"{nb} en backwardation, {len(carry_df) - nb} en contango.")
    except Exception as exc:  # noqa: BLE001
        print(f"      Carry live indisponible ({exc}) — "
              f"nécessite l'accès aux contrats listés (réseau).")

    print(f"\nTerminé. Livrables dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
