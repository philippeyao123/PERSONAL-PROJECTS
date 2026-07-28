"""Generate the README figures and vector figures used by the arXiv paper."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA, FIGS, REPORTS, ROOT

PAPER_FIGS = ROOT / "paper" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
C = {"actual": "#1a1a2e", "pred": "#e63946", "ridge": "#457b9d",
     "fv": "#e63946", "prompt": "#1d3557", "long": "#2a9d8f", "short": "#e76f51"}


def save(fig: plt.Figure, stem: str) -> None:
    """Write a compact README PNG and a vector PDF from the same figure."""
    fig.tight_layout()
    fig.savefig(FIGS / f"{stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGS / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


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
    save(fig, "01_price_history")


def fig_pred_vs_actual(res: pd.DataFrame) -> None:
    last = res.loc[res.index >= res.index.max() - pd.Timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(last.index, last["y_true"], lw=1.1, color=C["actual"], label="actual DA")
    ax.plot(last.index, last["lgbm"], lw=1.0, color=C["pred"], label="LightGBM (D-1)")
    ax.plot(last.index, last["naive_w"], lw=0.8, color="#a8a8a8", ls="--",
            label="naive (lag-168h)")
    ax.set_ylabel("EUR/MWh"); ax.legend(ncols=3, fontsize=8)
    ax.set_title("Out-of-sample next-day hourly forecasts -- last 14 days")
    save(fig, "02_pred_vs_actual")


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
    save(fig, "03_scatter")


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
    save(fig, "04_mae_by_hour")


def fig_importance(info: dict) -> None:
    imp = pd.Series(info["importances"]).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(imp.index, imp.values, color=C["ridge"])
    ax.set_title("LightGBM feature importance (top 12, split count)")
    save(fig, "05_feature_importance")


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
    save(fig, "06_fair_value_vs_prompt")


def fig_model_uncertainty() -> None:
    metrics = pd.read_csv(DATA / "model_comparison.csv").set_index("model")
    order = ["naive_w", "ridge", "lgbm"]
    labels = ["Weekly naive", "Ridge", "LightGBM"]
    frame = metrics.loc[order]
    yerr = np.vstack([
        frame["mae"] - frame["mae_ci_low"],
        frame["mae_ci_high"] - frame["mae"],
    ])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(
        labels, frame["mae"], yerr=yerr, capsize=4,
        color=["#9ca3af", C["ridge"], C["pred"]],
    )
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Full-year walk-forward error with day-block 95% intervals")
    save(fig, "07_model_uncertainty")


def fig_cumulative_skill(res: pd.DataFrame) -> None:
    days = res.index.tz_convert("Europe/Berlin").normalize()
    daily = pd.DataFrame(index=res.index)
    for model in ("naive_w", "ridge", "lgbm"):
        daily[model] = (res[model] - res["y_true"]).abs()
    daily = daily.groupby(days).mean()
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for comparator, color, label in (
        ("naive_w", "#6b7280", "LightGBM vs weekly naive"),
        ("ridge", C["ridge"], "LightGBM vs Ridge"),
    ):
        improvement = (daily[comparator] - daily["lgbm"]).expanding().mean()
        ax.plot(improvement.index, improvement, color=color, lw=1.5, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Cumulative mean MAE gain (EUR/MWh)")
    ax.set_title("Stability of forecast skill through the out-of-sample year")
    ax.legend(fontsize=8)
    save(fig, "08_cumulative_skill")


def fig_season_hour_heatmap(res: pd.DataFrame) -> None:
    local = res.index.tz_convert("Europe/Berlin")
    month = local.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"], default="Autumn",
    )
    frame = pd.DataFrame({
        "season": season,
        "hour": local.hour,
        "ae": (res["lgbm"] - res["y_true"]).abs().to_numpy(),
    })
    order = ["Winter", "Spring", "Summer", "Autumn"]
    matrix = frame.pivot_table(
        index="season", columns="hour", values="ae", aggfunc="mean"
    ).reindex(order)
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_xlabel("Delivery hour (CET/CEST)")
    ax.set_title("LightGBM MAE by season and delivery hour")
    fig.colorbar(image, ax=ax, label="MAE (EUR/MWh)", pad=0.02)
    save(fig, "09_season_hour_heatmap")


def fig_conformal_path() -> None:
    conformal = pd.read_csv(DATA / "conformal_diagnostics.csv", index_col=0)
    conformal.index = pd.to_datetime(conformal.index, utc=True)
    last = conformal.loc[
        conformal.index >= conformal.index.max() - pd.Timedelta(days=14)
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    ax.fill_between(
        last.index, last["lower"], last["upper"],
        color="#93c5fd", alpha=0.45, label="90% prequential interval",
    )
    ax.plot(last.index, last["y_true"], color=C["actual"], lw=1.0, label="actual")
    ax.plot(last.index, last["lgbm"], color=C["pred"], lw=0.9, label="forecast")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Strictly prequential conformal intervals -- last 14 days")
    ax.legend(ncols=3, fontsize=8)
    save(fig, "10_conformal_path")


def fig_conformal_coverage() -> None:
    conformal = pd.read_csv(DATA / "conformal_diagnostics.csv")
    coverage = conformal.groupby("hour")["covered"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = np.where(coverage >= 0.90, C["long"], C["short"])
    ax.bar(coverage.index, coverage.values, color=colors)
    ax.axhline(0.90, color="black", lw=0.9, ls="--", label="90% nominal")
    ax.set_ylim(0.75, 1.0)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Delivery hour (CET/CEST)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal coverage reveals daytime undercoverage")
    ax.legend(fontsize=8)
    save(fig, "11_conformal_coverage")


def fig_ablation() -> None:
    frame = pd.read_csv(DATA / "ablation_metrics.csv").sort_values("mae")
    labels = frame["specification"].str.replace("_", " ").str.title()
    yerr = np.vstack([
        frame["mae"] - frame["mae_ci_low"],
        frame["mae_ci_high"] - frame["mae"],
    ])
    fig, ax = plt.subplots(figsize=(7.7, 3.6))
    bars = ax.bar(
        labels, frame["mae"], yerr=yerr, capsize=3,
        color=[C["pred"]] + [C["ridge"]] * (len(frame) - 1),
    )
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Feature-family ablations under an identical walk-forward design")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "12_ablation")


def fig_signal_sensitivity() -> None:
    frame = pd.read_csv(DATA / "signal_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), sharex=True)
    colors = {5: "#457b9d", 7: "#e63946", 14: "#2a9d8f"}
    for window, group in frame.groupby("prompt_window"):
        label = f"{window}-day prompt proxy"
        axes[0].plot(
            group["threshold"], group["hit_rate"],
            marker="o", color=colors[window], label=label,
        )
        axes[1].plot(
            group["threshold"], group["average_captured"],
            marker="o", color=colors[window], label=label,
        )
    axes[0].set_ylabel("Directional hit rate")
    axes[0].set_ylim(0.45, 0.9)
    axes[1].set_ylabel("Average signed spread (EUR/MWh)")
    for ax in axes:
        ax.set_xlabel("Absolute z-score threshold")
    axes[0].set_title("Direction")
    axes[1].set_title("Magnitude")
    axes[0].legend(fontsize=7)
    fig.suptitle("Prompt-proxy signal sensitivity (descriptive, not tradable P&L)")
    save(fig, "13_signal_sensitivity")


def fig_error_quantiles(res: pd.DataFrame) -> None:
    probabilities = np.linspace(0.50, 0.99, 50)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for model, color, label in (
        ("naive_w", "#9ca3af", "Weekly naive"),
        ("ridge", C["ridge"], "Ridge"),
        ("lgbm", C["pred"], "LightGBM"),
    ):
        errors = (res[model] - res["y_true"]).abs()
        ax.plot(probabilities, errors.quantile(probabilities), color=color, label=label)
    ax.set_xlabel("Absolute-error quantile")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Forecast-error tails remain material despite mean skill")
    ax.legend(fontsize=8)
    save(fig, "14_error_quantiles")


def main() -> None:
    df, res, daily, info = load()
    fig_price_history(df)
    fig_pred_vs_actual(res)
    fig_scatter(res, info)
    fig_mae_by_hour(res)
    fig_importance(info)
    fig_fair_value(daily)
    fig_model_uncertainty()
    fig_cumulative_skill(res)
    fig_season_hour_heatmap(res)
    fig_conformal_path()
    fig_conformal_coverage()
    fig_ablation()
    fig_signal_sensitivity()
    fig_error_quantiles(res)
    print("figures written:", sorted(p.name for p in FIGS.glob("*.pdf")))


if __name__ == "__main__":
    main()
