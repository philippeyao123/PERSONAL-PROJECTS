"""Figure generation for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

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


def plot_iv_vs_rv(iv: pd.Series, fwd_rv: pd.Series, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    (iv * 100).plot(ax=ax, color=_NAVY, lw=1.2, label="Implied vol (VIX)")
    (fwd_rv * 100).plot(ax=ax, color=_TEAL, lw=1.0, alpha=0.8,
                        label="Forward realized vol (SPX)")
    ax.set_ylabel("Annualized vol (%)")
    ax.set_title("Implied vol systematically exceeds subsequent realized vol",
                 fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_equity_curves(timed: pd.Series, always_short: pd.Series, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    timed.plot(ax=ax, color=_NAVY, lw=2, label="Z-score timed")
    always_short.plot(ax=ax, color=_RUST, lw=1.5, alpha=0.8,
                      label="Always short variance")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative P&L (vol points)")
    ax.set_title("Short-vol carry vs crash drawdowns (2018, 2020)",
                 fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_return_distribution(pnl: pd.Series, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnl.dropna(), bins=60, color=_TEAL, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(pnl.mean(), color=_RUST, lw=1.5, ls="--",
               label=f"mean {pnl.mean():.3f}")
    ax.set_xlabel("Period P&L (vol points)")
    ax.set_ylabel("Frequency")
    ax.set_title("Short-variance P&L: positive carry, fat left tail",
                 fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_correlation(imp: pd.Series, real: pd.Series, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    imp.plot(ax=ax, color=_NAVY, lw=1.2, label="Implied correlation")
    real.plot(ax=ax, color=_TEAL, lw=1.0, alpha=0.8,
              label="Realized correlation")
    ax.set_ylabel("Average pairwise correlation")
    ax.set_title("Dispersion: implied vs realized correlation",
                 fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, path)
