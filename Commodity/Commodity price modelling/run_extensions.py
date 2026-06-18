"""
Extensions pipeline:
  A. Schwartz Model 3 (3 factors): stochastic convenience yield + Vasicek rates,
     Kalman MLE on the WTI curve; filtered convenience yield path.
  B. Calendar-spread strategy on the filtered chi factor — strict OOS protocol
     (params frozen on train window, causal filtering on test window).
  C. Leung-Li optimal entry/exit bands on chi's OU dynamics, compared with
     heuristic z-bands, both backtested on the same OOS window.
Figures 07-09 in outputs/, results appended to outputs/results_extensions.txt.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_wti_term_structure
from src.models.three_factor import fit_three_factor, model_curve_3f, load_short_rate
from src.models.optimal_bands import optimal_bands
from src.strategy_calendar import run_calendar_strategy

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
OUT = "outputs"
lines: list[str] = []


def log(msg: str):
    print(msg)
    lines.append(msg)


def main():
    log_f, tau = load_wti_term_structure("2024-06-01", n_contracts=8)
    log(f"Panel: {log_f.shape[0]} dates x {log_f.shape[1]} contracts")

    # ------------------------------------------------ A. three-factor model
    p3, fac3, vas = fit_three_factor(log_f, tau)
    log(f"\n[Vasicek on 13w T-bill] a={vas.kappa:.2f} | m={vas.mu:.2%} | "
        f"sigma_r={vas.sigma:.2%}")
    log(f"[Schwartz Model 3, Kalman MLE] {p3.summary()}")

    r_now = load_short_rate("2026-01-01").iloc[-1]
    t_last = tau.values[-1]
    order = np.argsort(t_last)
    fit3 = model_curve_3f(p3, fac3["lnS"].iloc[-1], fac3["delta"].iloc[-1],
                          r_now, t_last[order])
    obs = np.exp(log_f.values[-1])[order]
    rmse3 = np.sqrt(np.mean((np.log(obs) - np.log(fit3)) ** 2)) * 1e4
    log(f"    -> last-day curve RMSE: {rmse3:.0f}bp | "
        f"filtered convenience yield today: {fac3['delta'].iloc[-1]:+.1%}")

    # slope of the observed curve (front minus back, annualised) for comparison
    slope = (log_f.iloc[:, 0] * 0).copy()
    fcol, bcol = tau.mean().idxmin(), tau.mean().idxmax()
    slope = -(log_f[bcol] - log_f[fcol]) / (tau[bcol] - tau[fcol])

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(t_last[order], obs, "o-", label="observed")
    ax[0].plot(t_last[order], fit3, "s--", label="3-factor model")
    ax[0].set_xlabel("maturity (yrs)"); ax[0].set_ylabel("$/bbl")
    ax[0].set_title(f"WTI curve, Schwartz Model 3 — {log_f.index[-1].date()}")
    ax[0].legend()
    ax[1].plot(fac3.index, fac3["delta"], lw=0.9, label="filtered conv. yield δ")
    ax[1].plot(slope.index, slope, lw=0.7, alpha=0.6,
               label="curve slope (annualised)")
    ax[1].axhline(p3.alpha, ls="--", lw=0.8, c="k", label="α (P long-run)")
    ax[1].set_title("Convenience yield vs curve backwardation"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/07_three_factor.png"); plt.close(fig)

    # --------------------------------- B. chi-driven calendar spread (OOS)
    bt_z = run_calendar_strategy(log_f, tau, train_frac=0.5,
                                 entry=1.0, exit_=0.25, cost_bps=2.0)
    ps = bt_z.params_train
    log(f"\n[Train-window Schwartz-Smith] {ps.summary()}")
    log(f"[Calendar spread, z-bands ±1.0/0.25] {bt_z.summary()}")

    # --------------------------------------------- C. Leung-Li optimal bands
    # Costs must be expressed in chi units: the tradeable log spread moves
    # (e^{-k tau_f} - e^{-k tau_b}) per unit of chi, so a round-trip cost of
    # 2bp x 4 leg-trades on the spread maps to cost / loading_diff in chi.
    tau_f, tau_b = tau[fcol].mean(), tau[bcol].mean()
    loading = abs(np.exp(-ps.kappa * tau_f) - np.exp(-ps.kappa * tau_b))
    c_chi = (2e-4 * 2) / loading           # 2bp x 2 legs per side
    log(f"\n[Cost mapping] chi loading of the spread = {loading:.3f} "
        f"-> per-side cost in chi units = {c_chi:.4f}")

    bands = optimal_bands(kappa=ps.kappa, theta=0.0, sigma=ps.sigma_chi,
                          r=0.04, c_buy=c_chi, c_sell=c_chi)
    sd_stat = ps.sigma_chi / np.sqrt(2 * ps.kappa)
    log(f"[Leung-Li on chi] {bands.summary()}")
    log(f"    -> in stationary-std units: entry {bands.entry/sd_stat:+.2f}sd, "
        f"exit {bands.exit/sd_stat:+.2f}sd (chi stationary std = {sd_stat:.3f})")

    bt_ll = run_calendar_strategy(log_f, tau, train_frac=0.5,
                                  entry=abs(bands.entry), exit_=abs(bands.exit),
                                  band_mode="cross",
                                  bands_in_chi_units=True, cost_bps=2.0)
    log(f"[Calendar spread, Leung-Li bands] {bt_ll.summary()}")

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    ax[0].plot(bt_z.chi.index, bt_z.chi, lw=0.8, label="filtered chi (OOS, causal)")
    for lv, c, lab in [(bands.entry, "tab:green", "LL entry/exit"),
                       (bands.exit, "tab:green", None),
                       (-bands.entry, "tab:green", None), (-bands.exit, "tab:green", None)]:
        ax[0].axhline(lv, ls="--", lw=0.7, c=c, label=lab)
    ax[0].axhline(sd_stat, ls=":", lw=0.7, c="tab:red", label="z ±1sd heuristic")
    ax[0].axhline(-sd_stat, ls=":", lw=0.7, c="tab:red")
    ax[0].set_title("Short-term factor chi with trading bands"); ax[0].legend(ncol=2)
    ax[1].plot(bt_z.equity.index, bt_z.equity, lw=1.0, label="z-bands")
    ax[1].plot(bt_ll.equity.index, bt_ll.equity, lw=1.0, label="Leung-Li bands")
    ax[1].set_title("OOS equity curves — chi calendar-spread strategy (net 2bp/leg)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/08_calendar_strategy.png"); plt.close(fig)

    # band sensitivity of LL solution to costs (illustration of the point)
    costs = np.linspace(0.0, 0.02, 9)
    ent, ext = [], []
    for c in costs:
        b = optimal_bands(ps.kappa, 0.0, ps.sigma_chi, r=0.04, c_buy=c, c_sell=c)
        ent.append(b.entry); ext.append(b.exit)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(costs * 1e4, np.array(ent) / sd_stat, "o-", label="entry d*")
    ax.plot(costs * 1e4, np.array(ext) / sd_stat, "s-", label="exit b*")
    ax.set_xlabel("round-trip cost (bp of spread)")
    ax.set_ylabel("threshold (stationary sd units)")
    ax.set_title("Leung-Li bands widen with costs — z-score rules can't do this")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/09_optimal_bands.png"); plt.close(fig)

    with open(f"{OUT}/results_extensions.txt", "w") as f:
        f.write("\n".join(lines))
    log(f"\nDone. Figures 07-09 in {OUT}/")


if __name__ == "__main__":
    main()
