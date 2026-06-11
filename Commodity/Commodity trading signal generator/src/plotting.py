"""
Génération du tearsheet graphique (PNG haute résolution).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown_series, rolling_sharpe

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
})


def tearsheet(results: dict, attribution: pd.DataFrame,
              individual_signals: dict, sectors: dict,
              out_path: str) -> None:
    net, gross = results["net"], results["gross"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # 1. Equity curves net vs brut
    ax = axes[0, 0]
    (1 + net).cumprod().plot(ax=ax, label="Net", lw=1.4, color="navy")
    (1 + gross).cumprod().plot(ax=ax, label="Gross", lw=1.0,
                               color="steelblue", alpha=0.6)
    ax.set_title("Cumulative Performance (net vs gross of costs)")
    ax.set_yscale("log")
    ax.legend()

    # 2. Drawdown
    ax = axes[0, 1]
    dd = drawdown_series(net)
    dd.plot(ax=ax, color="firebrick", lw=1.0)
    ax.fill_between(dd.index, dd, 0, color="firebrick", alpha=0.25)
    ax.set_title("Drawdown")

    # 3. Rolling Sharpe 1Y
    ax = axes[1, 0]
    rs = rolling_sharpe(net)
    rs.plot(ax=ax, color="darkgreen", lw=1.2)
    ax.axhline(rs.mean(), ls="--", color="gray",
               label=f"Mean = {rs.mean():.2f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 1Y Sharpe Ratio")
    ax.legend()

    # 4. Attribution par signal (equity curves)
    ax = axes[1, 1]
    for col in attribution.columns:
        (1 + attribution[col]).cumprod().plot(ax=ax, lw=1.1, label=col)
    ax.set_title("Per-Signal Attribution (standalone backtests)")
    ax.legend(fontsize=8)

    # 5. Corrélation des signaux (PnL journaliers)
    ax = axes[2, 0]
    corr = attribution.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.annotate(f"{corr.iloc[i, j]:.2f}", (j, i),
                        ha="center", va="center", fontsize=8)
    ax.set_title("Signal PnL Correlation Matrix")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 6. PnL cumulé par secteur
    ax = axes[2, 1]
    asset_pnl = results["asset_pnl"]
    sector_pnl = asset_pnl.T.groupby(
        lambda c: sectors.get(c, "Other")).sum().T
    sector_pnl.cumsum().plot(ax=ax, lw=1.2)
    ax.set_title("Cumulative PnL by Sector")
    ax.legend(fontsize=8)

    fig.suptitle("Commodity Multi-Signal Strategy — Tearsheet",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] Tearsheet sauvegardé : {out_path}")


def current_signals_chart(composite: pd.DataFrame, universe: dict,
                          out_path: str) -> None:
    """Snapshot des signaux courants (dernière date) — vue desk."""
    last = composite.iloc[-1].dropna().sort_values()
    labels = [f"{universe.get(t, t)} ({t})" for t in last.index]
    colors = ["firebrick" if v < 0 else "darkgreen" for v in last.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, last.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title(f"Current Composite Signals — {composite.index[-1].date()}")
    ax.set_xlabel("Signal strength  [-1 = max short, +1 = max long]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] Snapshot signaux sauvegardé : {out_path}")


def carry_charts(carry_df, curves, universe: dict, out_path: str) -> None:
    """Deux panneaux : carry annualisé par actif + courbes à terme
    normalisées (F/F_near)."""
    import datetime as dt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    d = carry_df.sort_values("carry_annualized")
    labels = [f"{universe.get(t, t)}" for t in d.index]
    colors = ["firebrick" if v < 0 else "darkgreen"
              for v in d["carry_annualized"]]
    ax1.barh(labels, d["carry_annualized"], color=colors, alpha=0.85)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("Annualized Carry (12m seasonal pair)\n"
                  "green = backwardation (long), red = contango (short)")
    ax1.set_xlabel("ln(F_near / F_near+12m) / 1y")

    today = dt.date.today()
    for root, curve in curves.items():
        t = [(m - today).days / 365.25 for m in curve.index]
        ax2.plot(t, curve / curve.iloc[0], marker="o", ms=3, lw=1.1,
                 label=root)
    ax2.axhline(1.0, color="black", lw=0.8)
    ax2.set_title("Live Term Structures (normalized to nearest contract)")
    ax2.set_xlabel("Time to maturity (years)")
    ax2.set_ylabel("F(T) / F(near)")
    ax2.legend(fontsize=7, ncol=3)

    fig.suptitle(f"Commodity Carry — Live Snapshot ({today})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] Carry snapshot sauvegardé : {out_path}")
