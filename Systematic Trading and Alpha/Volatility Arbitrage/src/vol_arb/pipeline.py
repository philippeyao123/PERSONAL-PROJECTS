"""End-to-end pipeline for both strategies, with strict information sets.

Critical distinction the original notebook missed:
  - The TRADING SIGNAL at t may use only data observable at t: implied vol
    (VIX) and TRAILING realized vol. It must NOT use forward realized vol.
  - The P&L REALIZATION uses forward realized vol (what actually happens after
    t) — but that is the outcome, not an input to the decision.

So we build the signal from (IV_t - trailing_RV_t), and we settle the P&L
against forward RV. Conflating the two is what made the original notebook
circular.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from vol_arb.data.loader import (
    VolDataLoader,
    forward_realized_vol,
    realized_vol,
)
from vol_arb.diagnostics.metrics import performance_stats, vrp_summary
from vol_arb.signals.vrp import (
    implied_correlation,
    realized_correlation,
    variance_risk_premium,
    zscore,
)
from vol_arb.strategy.backtest import (
    CostParams,
    always_short,
    dispersion_pnl,
    variance_swap_pnl,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("vol_arb.pipeline")


def run_vrp(data: dict, horizon: int = 21, sample_step: int = 21) -> dict:
    """Variance risk premium strategy on SPX/VIX.

    Signal uses IV vs TRAILING RV (observable at t); P&L settles vs forward RV.
    P&L is sampled every `sample_step` days to avoid overlapping-window
    autocorrelation inflating the Sharpe.
    """
    core = data["core"]
    iv = core["VIX"] / 100.0
    trail_rv = realized_vol(core["SPX"], window=horizon)
    fwd_rv = forward_realized_vol(core["SPX"], horizon=horizon)

    # Tradable signal: premium using TRAILING rv (known at t).
    signal_premium = iv - trail_rv
    z = zscore(signal_premium, window=252)

    # Position from the observable signal.
    pos = pd.Series(0.0, index=z.index)
    pos[z > 0.5] = -1.0     # rich premium vs recent realized -> short vol
    pos[z < -0.5] = 1.0     # cheap/negative -> long vol

    # Settle P&L against forward RV (the realized outcome).
    bt = variance_swap_pnl(iv, fwd_rv, pos, CostParams())
    sampled = bt["pnl_net"].iloc[::sample_step].dropna()
    ppy = 252 / sample_step
    stats = performance_stats(sampled, periods_per_year=int(ppy))

    # Benchmark: always short, to expose the raw crash profile.
    bt_short = variance_swap_pnl(iv, fwd_rv, always_short(iv.index), CostParams())
    short_sampled = bt_short["pnl_net"].iloc[::sample_step].dropna()
    short_stats = performance_stats(short_sampled, periods_per_year=int(ppy))

    vrp_df = variance_risk_premium(iv, fwd_rv)
    return {
        "bt": bt, "stats": stats, "short_stats": short_stats,
        "summary": vrp_summary(vrp_df), "position": pos,
        "equity": bt["pnl_net"].iloc[::sample_step].cumsum(),
        "short_equity": bt_short["pnl_net"].iloc[::sample_step].cumsum(),
    }


def run_dispersion(data: dict, horizon: int = 21) -> dict:
    """Dispersion strategy: implied vs realized average correlation.

    Implied correlation is backed out from index IV and constituent IV proxies.
    Without a single-name IV history, we proxy constituent IV by a scaled
    trailing realized vol (a documented simplification); the *relationship*
    between implied and realized correlation is the object of study. Signal
    uses only data known at t (trailing realized correlation, current implied).
    """
    core, basket = data["core"], data["basket"]
    index_iv = core["VIX"] / 100.0

    # Equal weights (documented simplification vs true cap weights).
    weights = pd.Series(1.0, index=basket.columns)

    # Constituent IV proxy: trailing realized vol scaled by the index VRP ratio
    # (index IV / index trailing RV), so single-name "IV" carries a comparable
    # premium. Documented approximation in lieu of single-name option data.
    cons_rv = realized_vol(basket, window=horizon)
    spx_trail = realized_vol(core["SPX"], window=horizon)
    premium_ratio = (index_iv / spx_trail).clip(0.8, 2.5)
    cons_iv = cons_rv.mul(premium_ratio, axis=0)

    imp_corr = implied_correlation(index_iv, cons_iv, weights)
    basket_ret = np.log(basket / basket.shift(1))
    real_corr = realized_correlation(basket_ret, window=horizon)
    fwd_real_corr = real_corr.shift(-horizon)  # outcome after t

    # Signal: short correlation when implied > trailing realized (rich).
    aligned = pd.DataFrame({"ic": imp_corr, "rc": real_corr}).dropna()
    spread = aligned["ic"] - aligned["rc"]
    z = zscore(spread, window=126)
    pos = pd.Series(0.0, index=z.index)
    pos[z > 0.5] = -1.0   # rich implied corr -> short correlation
    pos[z < -0.5] = 1.0

    bt = dispersion_pnl(imp_corr, fwd_real_corr.reindex(imp_corr.index),
                        pos.reindex(imp_corr.index).fillna(0.0))
    sampled = bt["pnl_net"].iloc[::horizon].dropna()
    stats = (performance_stats(sampled, periods_per_year=int(252 / horizon))
             if len(sampled) > 3 else None)
    return {
        "bt": bt, "stats": stats, "implied_corr": imp_corr,
        "realized_corr": real_corr,
        "equity": bt["pnl_net"].iloc[::horizon].cumsum(),
    }


def main() -> None:
    data = VolDataLoader().load(use_cache=True)
    print("\n" + "=" * 60)
    print("VARIANCE RISK PREMIUM — SPX/VIX")
    print("=" * 60)
    vrp = run_vrp(data)
    s = vrp["summary"]
    print(f"IV exceeded subsequent RV {s['pct_iv_above_rv']:.1f}% of days; "
          f"mean premium {s['mean_vrp_vol_pts']:.2f} vol pts "
          f"(IV {s['mean_iv_pct']:.1f}% vs fwd RV {s['mean_fwd_rv_pct']:.1f}%).")
    st, sh = vrp["stats"], vrp["short_stats"]
    print(f"\n{'':22}{'Timed':>10}{'Always-short':>14}")
    for label, a, b in [
        ("Sharpe", st.sharpe, sh.sharpe),
        ("Skew", st.skew, sh.skew),
        ("Max drawdown", st.max_drawdown, sh.max_drawdown),
        ("Worst period", st.worst_day, sh.worst_day),
        ("CVaR 95%", st.cvar_95, sh.cvar_95),
    ]:
        print(f"{label:22}{a:>10.2f}{b:>14.2f}")
    print("\nThe always-short book shows the true short-vol signature: positive")
    print("average carry, strongly negative skew, deep crash drawdowns (2018,")
    print("2020). That asymmetry — not a headline Sharpe — is the real story.")

    print("\n" + "=" * 60)
    print("DISPERSION — implied vs realized correlation")
    print("=" * 60)
    disp = run_dispersion(data)
    if disp["stats"]:
        print(f"Mean implied corr {disp['implied_corr'].mean():.2f} vs "
              f"realized {disp['realized_corr'].mean():.2f}; "
              f"dispersion Sharpe {disp['stats'].sharpe:.2f}")


if __name__ == "__main__":
    main()
