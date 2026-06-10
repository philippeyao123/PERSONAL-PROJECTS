"""
Spike classification with walk-forward validation, two information sets:
  A. Day-ahead (forecast vintages only)
  B. Nowcast (adds wind forecast error, realized fundamentals, short lags)

Models: Logistic Regression (baseline), Random Forest, LightGBM.
Evaluation: expanding-window walk-forward, 6 folds x 2 months over the
final 12 months. PR-AUC is the headline metric (6% base rate makes
ROC-AUC flattering). SHAP on LightGBM. Simple DA-vs-imbalance P&L backtest.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, f1_score,
                             precision_score, recall_score)
import lightgbm as lgb
import shap, json, warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
FIG = "/home/claude/figs"

df = pd.read_parquet("/home/claude/data/dataset.parquet")

FEATURES_DA = [
    "nd_fcst_da", "tsd_fcst_da", "wind_fcst_da", "resid_demand_fcst", "renew_pen_fcst",
    "temp_gb", "rh_gb", "solar_gb", "wind_ms_offshore", "hdd", "cdd",
    "hour_sin", "hour_cos", "dow", "month", "is_weekend", "is_evening_peak",
    "doy_sin", "doy_cos",
    "sysprice_lag48", "sysprice_lag96", "sysprice_lag336",
    "niv_lag48", "sysprice_d1_mean", "sysprice_d1_max",
    "sysprice_w_std", "spike_rate_30d",
]
FEATURES_NC = FEATURES_DA + ["wind_fcst_err", "resid_demand_act", "renew_pen_act",
                             "sysprice_lag2", "niv_lag2"]

def models():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=0),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=60, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=5.0, scale_pos_weight=4, random_state=0, verbose=-1),
    }

# ---------------------------------------------------------------- walk-forward folds
test_start = df.index.max() - pd.DateOffset(months=12)
fold_edges = pd.date_range(test_start, df.index.max(), periods=7)

def walk_forward(features):
    preds = {name: pd.Series(dtype=float) for name in models()}
    y_all = pd.Series(dtype=float)
    d = df.dropna(subset=features)
    for i in range(6):
        tr = d[d.index < fold_edges[i]]
        te = d[(d.index >= fold_edges[i]) & (d.index < fold_edges[i + 1])]
        if te.empty:
            continue
        Xtr, ytr = tr[features], tr["spike"]
        Xte, yte = te[features], te["spike"]
        y_all = pd.concat([y_all, yte])
        for name, m in models().items():
            m.fit(Xtr, ytr)
            p = pd.Series(m.predict_proba(Xte)[:, 1], index=te.index)
            preds[name] = pd.concat([preds[name], p])
        print(f"  fold {i+1}/6: train {len(tr)}, test {len(te)}, "
              f"spikes {yte.mean():.1%}", flush=True)
    return y_all, preds

results = {}
for label, feats in [("day_ahead", FEATURES_DA), ("nowcast", FEATURES_NC)]:
    print(f"== {label}", flush=True)
    y, preds = walk_forward(feats)
    res = {}
    for name, p in preds.items():
        # threshold = max-F1 on first half of OOS, evaluated on second half
        n2 = len(p) // 2
        pr, rc, th = precision_recall_curve(y.iloc[:n2], p.iloc[:n2])
        f1s = 2 * pr * rc / (pr + rc + 1e-12)
        tau = th[np.argmax(f1s[:-1])]
        yh = (p.iloc[n2:] >= tau).astype(int)
        res[name] = {
            "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "f1": f1_score(y.iloc[n2:], yh),
            "precision": precision_score(y.iloc[n2:], yh),
            "recall": recall_score(y.iloc[n2:], yh),
            "tau": float(tau),
        }
    results[label] = {"y": y, "preds": preds, "metrics": res}
    print(pd.DataFrame(res).T.round(3))

base_rate = results["day_ahead"]["y"].mean()
metrics_out = {k: v["metrics"] for k, v in results.items()}
metrics_out["base_rate"] = float(base_rate)
json.dump(metrics_out, open("/home/claude/data/metrics.json", "w"), indent=2)

# ---------------------------------------------------------------- FIG 1: hockey stick
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
samp = df.sample(8000, random_state=0)
axes[0].scatter(samp["resid_demand_act"] / 1000, samp["system_price"],
                s=3, alpha=0.25, c="#1f4e79")
bins = pd.qcut(df["resid_demand_act"], 25)
med = df.groupby(bins, observed=True).agg(x=("resid_demand_act", "median"),
                                          p50=("system_price", "median"),
                                          p95=("system_price", lambda s: s.quantile(.95)))
axes[0].plot(med["x"] / 1000, med["p50"], c="orange", lw=2, label="median")
axes[0].plot(med["x"] / 1000, med["p95"], c="red", lw=2, ls="--", label="95th pct")
axes[0].set(xlabel="Residual demand (demand − wind − solar), GW",
            ylabel="System price (£/MWh)", ylim=(-100, 600),
            title="Non-linearity: price vs system tightness")
axes[0].legend()
axes[1].scatter(samp["renew_pen_act"], samp["system_price"], s=3, alpha=0.25, c="#1f4e79")
bins2 = pd.qcut(df["renew_pen_act"], 25)
med2 = df.groupby(bins2, observed=True).agg(x=("renew_pen_act", "median"),
                                            p95=("system_price", lambda s: s.quantile(.95)))
axes[1].plot(med2["x"], med2["p95"], c="red", lw=2, ls="--", label="95th pct")
axes[1].set(xlabel="Renewable penetration (wind+solar)/demand",
            ylabel="System price (£/MWh)", ylim=(-100, 600),
            title="Price vs renewable penetration")
axes[1].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/01_nonlinearity.png"); plt.close()

# ---------------------------------------------------------------- FIG 2: price series + spikes
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(df.index, df["system_price"], lw=0.3, c="#1f4e79")
sp = df[df["spike"] == 1]
ax.scatter(sp.index, sp["system_price"], s=2, c="red", label=f"spikes ({df['spike'].mean():.1%})")
ax.plot(df.index, df["spike_threshold"], lw=0.8, c="orange", label="rolling 90d p95 threshold")
ax.set(ylabel="System price (£/MWh)", title="GB imbalance (system) price, Jun 2024 – May 2026")
ax.legend(markerscale=4)
plt.tight_layout(); plt.savefig(f"{FIG}/02_price_series.png"); plt.close()

# ---------------------------------------------------------------- FIG 3: PR curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, label in zip(axes, ["day_ahead", "nowcast"]):
    y = results[label]["y"]
    for name, p in results[label]["preds"].items():
        pr, rc, _ = precision_recall_curve(y, p)
        ap = results[label]["metrics"][name]["pr_auc"]
        ax.plot(rc, pr, lw=1.5, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y.mean(), ls=":", c="gray", label=f"base rate {y.mean():.3f}")
    ax.set(xlabel="Recall", title=label.replace("_", "-"))
    ax.legend(fontsize=7)
axes[0].set_ylabel("Precision")
plt.suptitle("Out-of-sample precision–recall (walk-forward, 12 months)")
plt.tight_layout(); plt.savefig(f"{FIG}/03_pr_curves.png"); plt.close()

# ---------------------------------------------------------------- FIG 4: SHAP (LightGBM, DA)
tr = df[df.index < fold_edges[0]]
m = models()["LightGBM"].fit(tr[FEATURES_DA], tr["spike"])
te = df[df.index >= fold_edges[0]]
expl = shap.TreeExplainer(m)
Xs = te[FEATURES_DA].sample(3000, random_state=0)
sv = expl.shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else sv
shap.summary_plot(sv, Xs,
                  show=False, max_display=15)
plt.title("SHAP — LightGBM day-ahead model", fontsize=10)
plt.tight_layout(); plt.savefig(f"{FIG}/04_shap_da.png", bbox_inches="tight"); plt.close()

# nowcast SHAP
nc = df.dropna(subset=FEATURES_NC)
tr2, te2 = nc[nc.index < fold_edges[0]], nc[nc.index >= fold_edges[0]]
m2 = models()["LightGBM"].fit(tr2[FEATURES_NC], tr2["spike"])
Xs2 = te2[FEATURES_NC].sample(3000, random_state=0)
sv2 = shap.TreeExplainer(m2).shap_values(Xs2)
sv2 = sv2[1] if isinstance(sv2, list) else sv2
shap.summary_plot(sv2, Xs2,
                  show=False, max_display=15)
plt.title("SHAP — LightGBM nowcast model", fontsize=10)
plt.tight_layout(); plt.savefig(f"{FIG}/05_shap_nowcast.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- FIG 5: toy backtest (DA model)
# Strategy: when P(spike) >= tau at the day-ahead stage, buy 1 MWh day-ahead
# (proxy: MID price) and sell at the imbalance price. Long-only spike capture.
y = results["day_ahead"]["y"]
p = results["day_ahead"]["preds"]["LightGBM"]
tau = results["day_ahead"]["metrics"]["LightGBM"]["tau"]
bt = df.loc[p.index, ["system_price", "mid_price"]].copy()
bt["signal"] = (p >= tau).astype(int)
bt["pnl"] = bt["signal"] * (bt["system_price"] - bt["mid_price"])
daily = bt["pnl"].resample("D").sum()
trades = int(bt["signal"].sum())
hit = (bt.loc[bt["signal"] == 1, "pnl"] > 0).mean()
sharpe = daily.mean() / daily.std() * np.sqrt(365)
equity = daily.cumsum()
dd = (equity - equity.cummax()).min()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(equity.index, equity.values, c="#1f4e79")
ax.set(ylabel="Cumulative P&L (£/MWh traded)",
       title=f"Toy spike-capture strategy (long DA, sell imbalance) — "
             f"Sharpe {sharpe:.2f}, hit rate {hit:.0%}, {trades} trades, maxDD £{dd:.0f}")
plt.tight_layout(); plt.savefig(f"{FIG}/06_backtest.png"); plt.close()
json.dump({"sharpe": float(sharpe), "hit_rate": float(hit), "trades": trades,
           "total_pnl": float(daily.sum()), "max_dd": float(dd)},
          open("/home/claude/data/backtest.json", "w"), indent=2)
print("backtest:", f"Sharpe {sharpe:.2f}, hit {hit:.0%}, trades {trades}, total £{daily.sum():.0f}")
print("done")
