"""Figure generation for the ML option pricing project.

All figures are written to images/ and referenced from the README.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY = (
    "#2563eb", "#f59e0b", "#10b981", "#ef4444", "#6b7280")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- 1. pipeline
def fig_pipeline(path="images/pipeline_architecture.png"):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eff6ff", ec=C_BLUE):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4,
                                   zorder=2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.2, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=1.3))

    box(0.0, 2.6, 1.9, 1.0, "Yahoo Finance\ncall chains (bid/ask)")
    box(0.0, 1.0, 1.9, 1.0, "Quality filters\n2-sided quotes, spread<50%\nT>7d, 0.7<S/K<1.3")
    box(2.6, 1.8, 2.1, 1.2, "Features\nlog-moneyness, √T,\nATM-IV per expiry, r,\nBS(ATM) / K")
    box(5.3, 2.6, 2.0, 1.0, "Direct mode\ntarget = C/K", fc="#fffbeb", ec=C_ORANGE)
    box(5.3, 1.0, 2.0, 1.0, "Residual mode\ntarget = (C−BS)/K", fc="#fffbeb", ec=C_ORANGE)
    box(7.9, 1.8, 1.9, 1.2, "Models ×5\nLin / SVR / RF /\nGBM / MLP\nGroupKFold by expiry")
    box(10.3, 1.8, 1.9, 1.2, "No-arbitrage\nbounds + isotonic(K)\n→ metrics, SHAP", fc="#ecfdf5", ec=C_GREEN)

    arrow(0.95, 2.6, 0.95, 2.0)
    arrow(1.9, 1.7, 2.6, 2.2)
    arrow(4.7, 2.6, 5.3, 3.0)
    arrow(4.7, 2.2, 5.3, 1.6)
    arrow(7.3, 3.0, 7.9, 2.6)
    arrow(7.3, 1.5, 7.9, 2.0)
    arrow(9.8, 2.4, 10.3, 2.4)
    ax.set_xlim(-0.2, 12.4); ax.set_ylim(0.6, 4.0)
    ax.set_title("Pipeline — leakage-safe, homogeneity-normalised, no-arbitrage constrained",
                 fontsize=10, pad=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------- 2. model comparison
def fig_model_comparison(res: pd.DataFrame, path="images/model_comparison.png"):
    sub = res[res["mode"].str.contains("clean|baseline")].copy()
    sub["label"] = sub["model"] + "\n" + sub["mode"].str.replace(r"\[clean\]", "", regex=True)
    sub = sub.sort_values("rmse_price")
    colors = [C_RED if m == "baseline" else
              (C_GREEN if "residual" in m else C_BLUE) for m in sub["mode"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(sub["label"], sub["rmse_price"], color=colors)
    ax.set_ylabel("Test RMSE ($, price units)")
    ax.set_title("Held-out expiries — clean features (ATM-IV anchor, no per-option IV)")
    for p, v in zip(ax.patches, sub["rmse_price"]):
        ax.text(p.get_x() + p.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (C_RED, C_BLUE, C_GREEN)]
    ax.legend(handles, ["BS (ATM vol) baseline", "direct mode", "residual mode"],
              frameon=False, fontsize=8)
    plt.xticks(fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------- 3. vol surface + pricing error
def fig_surface_error(df_all: pd.DataFrame, test: pd.DataFrame,
                      best_px: np.ndarray, path="images/vol_surface_pricing_error.png"):
    fig = plt.figure(figsize=(11, 4.2))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_trisurf(np.log(df_all["S"] / df_all["strike"]), df_all["T"],
                     df_all["iv"], cmap="viridis", linewidth=0.1, alpha=0.95)
    ax1.set_xlabel("log-moneyness"); ax1.set_ylabel("T (yrs)"); ax1.set_zlabel("IV")
    ax1.set_title("Implied vol surface (market)", fontsize=9)
    ax1.view_init(22, -55)

    ax2 = fig.add_subplot(1, 2, 2)
    bs_err = test["bs_atm"].values - test["mid"].values
    ml_err = best_px - test["mid"].values
    m = test["log_moneyness"].values
    ax2.scatter(m, bs_err, s=12, alpha=0.55, color=C_RED, label="BS(ATM vol) error")
    ax2.scatter(m, ml_err, s=12, alpha=0.55, color=C_GREEN, label="Best ML (residual) error")
    ax2.axhline(0, color=C_GREY, lw=0.8)
    ax2.set_xlabel("log-moneyness"); ax2.set_ylabel("pred − mid ($)")
    ax2.set_title("Pricing error vs moneyness — held-out expiries", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# -------------------------------------------------------- 4. SHAP importance
def fig_shap(shap_values, features, path="images/shap_importance.png"):
    imp = pd.Series(np.abs(shap_values).mean(0), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(imp.index, imp.values, color=C_BLUE)
    ax.set_xlabel("mean |SHAP| (contribution to (C−BS)/K)")
    ax.set_title("Feature importance — GBM, residual mode, clean features")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------- 5. BS residuals
def fig_residuals(test: pd.DataFrame, best_px: np.ndarray,
                  path="images/bs_residuals.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    bs_res = (test["mid"] - test["bs_atm"]).values
    ml_res = (test["mid"].values - best_px)

    sc = axes[0].scatter(test["log_moneyness"], bs_res, c=test["T"],
                         cmap="plasma", s=14, alpha=0.8)
    axes[0].axhline(0, color=C_GREY, lw=0.8)
    axes[0].set_xlabel("log-moneyness"); axes[0].set_ylabel("mid − BS(ATM) ($)")
    axes[0].set_title("Systematic BS mispricing = the smile (colour: T)", fontsize=9)
    plt.colorbar(sc, ax=axes[0], label="T (yrs)")

    bins = np.linspace(min(bs_res.min(), ml_res.min()),
                       max(bs_res.max(), ml_res.max()), 45)
    axes[1].hist(bs_res, bins=bins, alpha=0.6, color=C_RED, label="before (BS ATM)")
    axes[1].hist(ml_res, bins=bins, alpha=0.7, color=C_GREEN, label="after ML correction")
    axes[1].set_xlabel("residual ($)"); axes[1].set_ylabel("count")
    axes[1].set_title("Residual distribution — before vs after", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------ 6. greeks
def fig_greeks(test: pd.DataFrame, fitted: dict, feats,
               path="images/greeks_comparison.png"):
    """FD delta from ML surfaces vs analytical BS delta across moneyness."""
    from greeks_mc import fd_greeks, bs_delta
    S, K, T, r, iv = (test[c].values for c in ["S", "strike", "T", "r", "iv"])
    m = test["log_moneyness"].values
    order = np.argsort(m)
    d_ref = bs_delta(S, K, T, r, iv)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, (_, g) in enumerate(test.groupby("expiry")):
        gm = g["log_moneyness"].values
        o = np.argsort(gm)
        d_g = bs_delta(g["S"].values, g["strike"].values, g["T"].values,
                       g["r"].values, g["iv"].values)
        ax.plot(gm[o], d_g[o], color="black", lw=1.5, alpha=0.85,
                label="Analytical BS delta (per-option IV)" if i == 0 else None,
                zorder=5)
    palette = [C_BLUE, C_GREEN, C_ORANGE, C_RED, "#8b5cf6"]
    for (name, model), c in zip(fitted.items(), palette):
        d, _ = fd_greeks(test, model, feats, mode="residual")
        ax.scatter(m, d, s=12, alpha=0.65, color=c, label=f"{name} (FD)")
    ax.set_xlabel("log-moneyness"); ax.set_ylabel("delta")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Greeks via finite differences on the ML surface — "
                 "smoothness matters more than RMSE")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
