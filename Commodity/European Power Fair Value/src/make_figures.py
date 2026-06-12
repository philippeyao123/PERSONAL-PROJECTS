"""Generate all figures for the README / case-study document."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA, FIGS, REPORTS

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
C = {"actual": "#1a1a2e", "pred": "#e63946", "ridge": "#457b9d",
     "fv": "#e63946", "prompt": "#1d3557", "long": "#2a9d8f", "short": "#e76f51"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(DATA / "dataset.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    res = pd.read_csv(DATA / "predictions_oos.csv", index_col=0)
    res.index = pd.to_datetime(res.index, utc=True)
    daily = pd.read_csv(DATA / "daily_views.csv", index_col=0)
    daily.index = pd.to_datetime(daily.index, utc=True)
    info = json.loads((REPORTS / "model_metrics.json").read_text())
    return df, res, daily, info


def fig_price_history(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    d = df["price"].resample("D").mean()
    ax.plot(d.index, d, lw=0.8, color=C["actual"])
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    neg = df["price"].resample("D").apply(lambda s: (s < 0).sum())
    ax2 = ax.twinx()
    ax2.bar(neg.index, neg, color="#f4a261", alpha=0.5, width=1.0)
    ax2.set_ylabel("negative hours / day", color="#b25b1e")
    ax2.set_ylim(0, 24); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("DE-LU day-ahead baseload (daily mean) and negative-price hours")
    fig.tight_layout(); fig.savefig(FIGS / "01_price_history.png"); plt.close(fig)


def fig_pred_vs_actual(res: pd.DataFrame) -> None:
    last = res.loc[res.index >= res.index.max() - pd.Timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(last.index, last["y_true"], lw=1.1, color=C["actual"], label="actual DA")
    ax.plot(last.index, last["lgbm"], lw=1.0, color=C["pred"], label="LightGBM (D-1)")
    ax.plot(last.index, last["naive_w"], lw=0.8, color="#a8a8a8", ls="--",
            label="naive (lag-168h)")
    ax.set_ylabel("EUR/MWh"); ax.legend(ncols=3, fontsize=8)
    ax.set_title("Out-of-sample next-day hourly forecasts -- last 14 days")
    fig.tight_layout(); fig.savefig(FIGS / "02_pred_vs_actual.png"); plt.close(fig)


def fig_scatter(res: pd.DataFrame, info: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(res["y_true"], res["lgbm"], s=3, alpha=0.25, color=C["pred"],
               edgecolors="none")
    lim = np.percentile(res[["y_true", "lgbm"]].values, [0.2, 99.8])
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    m = info["metrics"]["lgbm"]
    ax.set_xlabel("actual (EUR/MWh)"); ax.set_ylabel("predicted (EUR/MWh)")
    ax.set_title(f"LightGBM OOS: MAE {m['MAE']} | R2 {m['R2']}")
    fig.tight_layout(); fig.savefig(FIGS / "03_scatter.png"); plt.close(fig)


def fig_mae_by_hour(res: pd.DataFrame) -> None:
    hod = res.index.tz_convert("Europe/Berlin").hour
    g = res.assign(h=hod).groupby("h").apply(
        lambda x: pd.Series({
            "lgbm": (x["y_true"] - x["lgbm"]).abs().mean(),
            "ridge": (x["y_true"] - x["ridge"]).abs().mean(),
            "naive": (x["y_true"] - x["naive_w"]).abs().mean()}),
        include_groups=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(g.index, g["naive"], color="#a8a8a8", ls="--", label="naive")
    ax.plot(g.index, g["ridge"], color=C["ridge"], label="ridge")
    ax.plot(g.index, g["lgbm"], color=C["pred"], lw=1.6, label="lgbm")
    ax.set_xlabel("hour (CET/CEST)"); ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Forecast error by delivery hour (errors peak at the evening ramp)")
    ax.legend(fontsize=8); ax.set_xticks(range(0, 24, 2))
    fig.tight_layout(); fig.savefig(FIGS / "04_mae_by_hour.png"); plt.close(fig)


def fig_importance(info: dict) -> None:
    imp = pd.Series(info["importances"]).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(imp.index, imp.values, color=C["ridge"])
    ax.set_title("LightGBM feature importance (top 12, split count)")
    fig.tight_layout(); fig.savefig(FIGS / "05_feature_importance.png"); plt.close(fig)


def fig_fair_value(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                             height_ratios=[2.2, 1])
    ax = axes[0]
    ax.plot(daily.index, daily["realised"], lw=0.8, color="#a8a8a8",
            label="realised DA baseload")
    ax.plot(daily.index, daily["fair_value"], lw=1.2, color=C["fv"],
            label="model fair value (D-1)")
    ax.plot(daily.index, daily["prompt_proxy"], lw=1.2, color=C["prompt"],
            label="prompt proxy (trailing 7d)")
    for view, col in (("LONG prompt", C["long"]), ("SHORT prompt", C["short"])):
        m = daily["view"] == view
        ax.scatter(daily.index[m], daily.loc[m, "fair_value"], s=14, color=col,
                   zorder=5, label=view)
    ax.set_ylabel("EUR/MWh"); ax.legend(ncols=5, fontsize=7.5)
    ax.set_title("Fair value vs prompt proxy and generated curve views (OOS)")
    ax2 = axes[1]
    ax2.bar(daily.index, daily["gap_z"], width=1.0,
            color=np.where(daily["gap_z"] > 0, C["long"], C["short"]), alpha=0.8)
    ax2.axhline(0.75, color="k", lw=0.6, ls="--"); ax2.axhline(-0.75, color="k", lw=0.6, ls="--")
    ax2.set_ylabel("gap z-score")
    fig.tight_layout(); fig.savefig(FIGS / "06_fair_value_vs_prompt.png"); plt.close(fig)


def main() -> None:
    df, res, daily, info = load()
    fig_price_history(df)
    fig_pred_vs_actual(res)
    fig_scatter(res, info)
    fig_mae_by_hour(res)
    fig_importance(info)
    fig_fair_value(daily)
    print("figures written:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
