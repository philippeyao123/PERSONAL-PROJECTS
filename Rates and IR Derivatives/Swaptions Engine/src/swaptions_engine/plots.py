"""Figure generation for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_NAVY = "#1f2a44"
_TEAL = "#2a9d8f"
_RUST = "#c1440e"
_GREY = "#8a8f98"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration_fit(fit_table, rmse, path):
    labels = [f"{int(r['expiry'])}x{int(r['tenor'])}" for r in fit_table]
    mkt = [r["market_vol_bp"] for r in fit_table]
    mdl = [r["model_vol_bp"] for r in fit_table]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, mkt, "o-", color=_NAVY, label="Market", lw=2, ms=6)
    ax.plot(x, mdl, "s--", color=_TEAL, label="G2++ model", lw=2, ms=5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ATM normal vol (bp)")
    ax.set_title(f"G2++ calibration to swaption surface (RMSE {rmse:.2f} bp)",
                 fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_bermudan_premium(n_exercises, prices, euro_price, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_exercises, [p * 1e4 for p in prices], "o-", color=_NAVY,
            lw=2, ms=6, label="Bermudan")
    ax.axhline(euro_price * 1e4, color=_RUST, ls="--", lw=1.5,
               label="European (1 exercise)")
    ax.set_xlabel("Number of exercise dates")
    ax.set_ylabel("Price (bp of notional)")
    ax.set_title("Bermudan early-exercise premium grows with exercise dates",
                 fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_bucketed_dv01(buckets, path):
    times = list(buckets.keys())
    vals = list(buckets.values())
    colors = [_TEAL if v >= 0 else _RUST for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(t) for t in times], vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Curve pillar (years)")
    ax.set_ylabel("DV01 ($ per bp)")
    ax.set_title("Key-rate DV01 profile (2Y5Y payer swaption)",
                 fontweight="bold")
    return _save(fig, path)
