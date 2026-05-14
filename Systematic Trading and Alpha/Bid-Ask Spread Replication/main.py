"""
Main Pipeline — Bid-Ask Replication & Optimal Quoting
=======================================================
Orchestrates all 6 modules end-to-end.

Usage:
    python main.py

Output:
    - dashboard.html (open in browser)
    - Console summary of all intermediate results

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import sys
import numpy as np
import pandas as pd

print("=" * 65)
print("  BID-ASK REPLICATION PIPELINE")
print("  Philippe-Emmanuel Yao · MSc Financial Mathematics, LSE")
print("=" * 65)

# ============================================================================
# MODULE 1 — Data Generation & Lee-Ready Classification
# ============================================================================
print("\n[M1] Data generation & Lee-Ready classification...")

from module1_data import simulate_tick_data, lee_ready_classify, classification_accuracy, MarketConfig

cfg = MarketConfig(seed=42)
df = simulate_tick_data(cfg)
df['lr_direction'] = lee_ready_classify(df)
acc = classification_accuracy(df)

print(f"    Trades          : {len(df):,}")
print(f"    Lee-Ready acc.  : {acc['overall_accuracy']:.1%}")
print(f"    Informed acc.   : {acc['informed_accuracy']:.1%}")
print(f"    Uninformed acc. : {acc['uninformed_accuracy']:.1%}")

# ============================================================================
# MODULE 2 — Spread Reconstruction
# ============================================================================
print("\n[M2] Bid-ask reconstruction (Roll, Corwin-Schultz, Level-2)...")

from module2_reconstruction import compare_estimators, level2_proxy

df_rec = compare_estimators(df)
l2_full = level2_proxy(df)

print("    Estimator MAE vs true spread:")
for col, label in [('roll_spread_error', 'Roll'), ('cs_spread_error', 'Corwin-Schultz'), ('l2_spread_error', 'Level-2')]:
    if col in df_rec.columns:
        mae = df_rec[col].abs().mean()
        corr = df_rec[[col.replace('_error',''), 'true_spread']].dropna().corr().iloc[0,1]
        print(f"      {label:<18}: MAE={mae:.5f}  Corr={corr:.3f}")

# ============================================================================
# MODULE 3 — Microstructure Features
# ============================================================================
print("\n[M3] Computing microstructure features (VPIN, OFI, Kyle λ, RV)...")

from module3_features import build_feature_matrix

feats = build_feature_matrix(df)
print(f"    Feature matrix   : {feats.shape}")
for c in ['vpin', 'ofi', 'kyle_lambda', 'realized_vol']:
    if c in feats.columns:
        corr = feats[[c, 'spread']].dropna().corr().iloc[0, 1]
        print(f"    Corr(spread, {c:<16}) = {corr:+.3f}")

# ============================================================================
# MODULE 4 — Spread Modelling
# ============================================================================
print("\n[M4] Spread modelling (OLS, Ridge, GBM) with walk-forward CV...")

from module4_modelling import train_spread_models

l2_series = l2_full.set_index('bucket_time')['spread_proxy']
res = train_spread_models(feats, l2_series)

print("    Walk-forward CV results (mean across folds):")
cv_agg = res.cv_results.groupby('model')[['mae', 'rmse', 'r2']].mean()
print(cv_agg.round(5).to_string())

# Best model predictions
best_model = cv_agg['r2'].idxmax()
print(f"\n    Best model : {best_model} (R² = {cv_agg.loc[best_model, 'r2']:.3f})")

if best_model == 'GBM':
    print("\n    GBM Feature Importances:")
    fi = res.models['GBM'].feature_importance(res.feature_names)
    for _, row in fi.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"      {row['feature']:<25} {bar} {row['importance']:.3f}")

# ============================================================================
# MODULE 5 — Optimal Quoting Backtest
# ============================================================================
print("\n[M5] Avellaneda-Stoikov backtest...")

from module5_quoting import run_backtest, summarise_backtest, ASParams

# Align predictions to tick level
n_test = len(res.X_test)
n_ticks = min(6000, len(df))
df_bt = df.head(n_ticks).copy()

# Use GBM predictions mapped to ticks (simplified: interpolate from test set)
pred_spread = np.interp(
    np.linspace(0, 1, n_ticks),
    np.linspace(0, 1, len(res.predictions[best_model])),
    np.maximum(res.predictions[best_model], 0)
)
vpin_vals = feats['vpin'].reindex(df_bt.index).fillna(0.5).values

bt = run_backtest(
    df_bt,
    predicted_spreads=pred_spread,
    vpins=vpin_vals,
    params=ASParams(gamma=0.1, k=1.5, max_inventory=50),
)

summary = summarise_backtest(bt)
print(f"    Total P&L         : {summary['total_pnl']:+.2f}")
print(f"    Sharpe Ratio      : {summary['sharpe_ratio']:.3f}")
print(f"    Max Drawdown      : {summary['max_drawdown']:.2f}")
print(f"    Total Fills       : {summary['total_fills']:,}")
print(f"    Avg |Inventory|   : {summary['avg_abs_inventory']:.1f}")
print(f"    Spread Income     : {summary['spread_income_total']:.2f}")
print(f"    Inventory PnL     : {summary['inventory_pnl_total']:.2f}")

# ============================================================================
# MODULE 6 — Dashboard
# ============================================================================
print("\n[M6] Generating dashboard...")

from module6_dashboard import generate_dashboard

# Prepare reconstruction df with timestamps
df_rec2 = df_rec.copy()
if not hasattr(df_rec2.index, 'strftime'):
    df_rec2.index = pd.date_range('2026-01-15', periods=len(df_rec2), freq='30s')

output = generate_dashboard(
    df_ticks=df_bt,
    df_reconstruction=df_rec2.reset_index().rename(columns={'index':'ts'}),
    df_features=feats.head(3000),
    df_backtest=bt,
    cv_results=res.cv_results,
    output_path='/mnt/user-data/outputs/bid_ask_dashboard.html',
)

print(f"\n{'=' * 65}")
print(f"  PIPELINE COMPLETE")
print(f"  Dashboard: {output}")
print(f"{'=' * 65}\n")
