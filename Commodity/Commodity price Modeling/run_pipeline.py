"""
End-to-end pipeline:
  1. Load WTI / Brent / NatGas front contracts + WTI futures curve panel.
  2. Fit Fourier seasonality on NatGas; OU on deseasonalised log price.
  3. Fit OU (Schwartz 1-factor) on WTI log spot.
  4. Fit Merton jump-diffusion on WTI daily returns.
  5. Calibrate Schwartz-Smith 2-factor on the WTI futures curve (Kalman MLE).
  6. Monte Carlo 1y spot distribution under the calibrated 2-factor model.
  7. Backtest WTI-Brent spread stat-arb with risk metrics.
Outputs: figures in outputs/, parameter summary in outputs/results.txt.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_front_contracts, load_wti_term_structure
from src.models.ou import fit_ou
from src.models.seasonality import fit_seasonality
from src.models.jumps import fit_merton, merton_density
from src.models.schwartz_smith import fit_schwartz_smith, model_curve, simulate_ss
from src.strategy import backtest_spread

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
OUT = "outputs"
log_lines: list[str] = []


def log(msg: str):
    print(msg)
    log_lines.append(msg)


def main():
    # ------------------------------------------------------------------ data
    px = load_front_contracts("2015-01-01")
    log(f"Front contracts: {px.index[0].date()} -> {px.index[-1].date()} ({len(px)} obs)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(px.index, px["WTI"], lw=0.8, label="WTI")
    ax[0].plot(px.index, px["Brent"], lw=0.8, label="Brent")
    ax[0].set_title("Crude oil front month ($/bbl)"); ax[0].legend()
    ax[1].plot(px.index, px["NatGas"], lw=0.8, color="tab:green")
    ax[1].set_title("Henry Hub natural gas ($/MMBtu)")
    fig.tight_layout(); fig.savefig(f"{OUT}/01_prices.png"); plt.close(fig)

    # -------------------------------------------------- seasonality (NatGas)
    ng_log = np.log(px["NatGas"])
    seas = fit_seasonality(ng_log, K=2)
    log(f"\n[NatGas seasonality] Fourier K=2, share of detrended variance: "
        f"{seas.r2_seasonal:.1%}")
    ou_ng = fit_ou(seas.residual)
    log(f"[NatGas deseasonalised OU] {ou_ng.summary()}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(ng_log.index, ng_log, lw=0.6, label="log NG")
    ax[0].plot(seas.fitted.index, seas.fitted, lw=1.2, label="trend + seasonality")
    ax[0].set_title("NatGas: Fourier seasonality fit"); ax[0].legend()
    month_avg = seas.seasonal.groupby(seas.seasonal.index.month).mean()
    ax[1].bar(month_avg.index, month_avg.values, color="tab:green")
    ax[1].set_title("Average seasonal component by month (log)")
    ax[1].set_xticks(range(1, 13))
    fig.tight_layout(); fig.savefig(f"{OUT}/02_seasonality.png"); plt.close(fig)

    # ------------------------------------------------------- OU on WTI spot
    ou_wti = fit_ou(np.log(px["WTI"]))
    log(f"\n[WTI 1-factor OU] {ou_wti.summary()}")

    # -------------------------------------------------- Merton jumps on WTI
    r_wti = np.log(px["WTI"]).diff().dropna()
    mjd = fit_merton(r_wti)
    log(f"[WTI Merton jump-diffusion] {mjd.summary()}")
    log(f"    -> diffusive vol {mjd.sigma:.1%} vs unconditional {r_wti.std()*np.sqrt(252):.1%}; "
        f"jumps carry the tails")

    grid = np.linspace(r_wti.quantile(0.001), r_wti.quantile(0.999), 400)
    from scipy.stats import norm
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(r_wti, bins=150, density=True, alpha=0.45, label="WTI daily returns")
    ax.plot(grid, merton_density(mjd, grid), lw=1.4, label="Merton MLE fit")
    ax.plot(grid, norm.pdf(grid, r_wti.mean(), r_wti.std()), "--", lw=1.1,
            label="Gaussian")
    ax.set_yscale("log"); ax.set_ylim(1e-3, None)
    ax.set_title("WTI return density (log scale): jumps capture the tails")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/03_jumps.png"); plt.close(fig)

    # ------------------------------------- Schwartz-Smith on futures curve
    log_f, tau = load_wti_term_structure("2024-06-01", n_contracts=8)
    log(f"\n[Term structure panel] {log_f.shape[0]} dates x {log_f.shape[1]} contracts "
        f"({log_f.index[0].date()} -> {log_f.index[-1].date()})")
    ss, factors = fit_schwartz_smith(log_f, tau)
    log(f"[Schwartz-Smith 2-factor, Kalman MLE] {ss.summary()}")

    last = -1
    tau_last = tau.values[last]
    order = np.argsort(tau_last)
    obs_curve = np.exp(log_f.values[last])[order]
    fit_curve = model_curve(ss, factors["chi"].iloc[last], factors["xi"].iloc[last],
                            tau_last[order])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(tau_last[order], obs_curve, "o-", label="observed")
    ax[0].plot(tau_last[order], fit_curve, "s--", label="model")
    ax[0].set_xlabel("maturity (yrs)"); ax[0].set_ylabel("$/bbl")
    ax[0].set_title(f"WTI futures curve fit — {log_f.index[last].date()}")
    ax[0].legend()
    ax[1].plot(factors.index, factors["chi"], lw=0.9, label="chi (short-term)")
    ax[1].plot(factors.index, factors["xi"], lw=0.9, label="xi (long-term, log)")
    ax[1].set_title("Filtered state factors"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/04_schwartz_smith.png"); plt.close(fig)

    rmse_bp = np.sqrt(np.mean((log_f.values[last][order] - np.log(fit_curve)) ** 2)) * 1e4
    log(f"    -> last-day curve RMSE: {rmse_bp:.0f}bp of log price")

    # --------------------------------------------------------- Monte Carlo
    paths = simulate_ss(ss, factors["chi"].iloc[-1], factors["xi"].iloc[-1],
                        n_days=252, n_paths=10_000)
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    log(f"\n[Monte Carlo 1y, 10k paths] spot now ≈ {paths[0,0]:.1f} | "
        f"median 1y: {q[2,-1]:.1f} | 90% CI 1y: [{q[0,-1]:.1f}, {q[4,-1]:.1f}]")

    t_axis = np.arange(paths.shape[0]) / 252
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.fill_between(t_axis, q[0], q[4], alpha=0.2, label="5-95%")
    ax.fill_between(t_axis, q[1], q[3], alpha=0.3, label="25-75%")
    ax.plot(t_axis, q[2], lw=1.3, label="median")
    for i in range(12):
        ax.plot(t_axis, paths[:, i], lw=0.4, alpha=0.5, color="grey")
    ax.set_xlabel("years"); ax.set_ylabel("$/bbl")
    ax.set_title("WTI spot — Schwartz-Smith Monte Carlo fan (10k paths)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/05_montecarlo.png"); plt.close(fig)

    # ------------------------------------------------------------ strategy
    bt = backtest_spread(px, z_in=2.0, z_out=0.5, cost_bps=2.0)
    log(f"\n[WTI-Brent spread stat-arb, base config] {bt.summary()}")

    log("\n[Sensitivity grid] Sharpe net of 2bp costs:")
    log("        z_window=40   z_window=60   z_window=90")
    for z_in_g in (1.5, 2.0, 2.5):
        row = []
        for zw in (40, 60, 90):
            b = backtest_spread(px, z_window=zw, z_in=z_in_g, z_out=0.5, cost_bps=2.0)
            row.append(f"{b.stats['sharpe']:+.2f}")
        log(f"z_in={z_in_g:.1f}     {row[0]}         {row[1]}         {row[2]}")

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1]})
    ax[0].plot(bt.equity.index, bt.equity, lw=1.0)
    ax[0].set_title("Equity curve — WTI/Brent spread (net of 2bp costs)")
    dd = bt.equity / bt.equity.cummax() - 1
    ax[1].fill_between(dd.index, dd, 0, color="tab:red", alpha=0.5)
    ax[1].set_title("Drawdown")
    ax[2].plot(bt.zscore.index, bt.zscore, lw=0.5)
    ax[2].axhline(2, ls="--", c="k", lw=0.6); ax[2].axhline(-2, ls="--", c="k", lw=0.6)
    ax[2].set_title("Spread z-score (entry ±2.0, exit ±0.5)")
    fig.tight_layout(); fig.savefig(f"{OUT}/06_strategy.png"); plt.close(fig)

    with open(f"{OUT}/results.txt", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nDone. Figures + results in {OUT}/")


if __name__ == "__main__":
    main()
