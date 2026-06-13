"""Figure generation for the README and reports.

Produces publication-quality PNGs from backtest / replication outputs:
    - equity curve (gross vs net)
    - factor IC bar chart with significance
    - TSMOM per-decade Sharpe decay
    - capacity curve (net return vs AUM)

Kept dependency-light (matplotlib only) and headless-safe (Agg backend).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# A restrained, professional palette.
_NAVY = "#1f2a44"
_TEAL = "#2a9d8f"
_RUST = "#c1440e"
_GREY = "#8a8f98"
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curve(
    gross: pd.Series, net: pd.Series, path: str | Path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    (1 + gross).cumprod().plot(ax=ax, color=_GREY, lw=1.5, label="Gross")
    (1 + net).cumprod().plot(ax=ax, color=_NAVY, lw=2.0, label="Net of costs")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_factor_ic(ic_summary: pd.DataFrame, path: str | Path) -> Path:
    df = ic_summary.sort_values("ic_mean")
    colors = [_TEAL if t > 2 else (_RUST if t < -2 else _GREY)
              for t in df["t_stat"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df.index, df["ic_mean"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Factor information coefficient (mean Rank-IC)",
                 fontweight="bold")
    ax.set_xlabel("Mean IC  (teal: t>2, rust: t<-2, grey: insignificant)")
    return _save(fig, path)


def plot_tsmom_decay(by_period: pd.DataFrame, path: str | Path) -> Path:
    decades = by_period[by_period["period"].str.endswith("s")]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_TEAL if s > 0.5 else (_RUST if s < 0.25 else _GREY)
              for s in decades["sharpe"]]
    ax.bar(decades["period"], decades["sharpe"], color=colors, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Time-Series Momentum: Sharpe decay by decade",
                 fontweight="bold")
    ax.set_ylabel("Net Sharpe ratio")
    for i, (_, row) in enumerate(decades.iterrows()):
        ax.text(i, row["sharpe"] + 0.03, f"{row['sharpe']:.2f}",
                ha="center", fontsize=9, fontweight="bold")
    return _save(fig, path)


def plot_capacity(capacity: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(capacity["aum"] / 1e9, capacity["net_return"] * 100,
            color=_NAVY, lw=2)
    ax.axhline(0, color=_RUST, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_title("Capacity: net return vs AUM", fontweight="bold")
    ax.set_xlabel("AUM ($B, log scale)")
    ax.set_ylabel("Net annual return (%)")
    return _save(fig, path)
