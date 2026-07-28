"""Forecasting models and walk-forward validation (Option A: next-day hourly).

Models
------
  naive_w   : same hour, same weekday last week (price_lag168)   [baseline 1]
  ridge     : standardised linear model                          [baseline 2]
  lgbm      : LightGBM gradient boosting                         [improved]

Validation
----------
Walk-forward over the last TEST_DAYS delivery days. Models are refit every
RETRAIN_EVERY days on an expanding window ending 24h before the first
predicted hour, then predict whole delivery days ahead -- exactly the cadence
of a live D-1 process. Metrics: MAE, RMSE, R^2, and skill vs the naive
baseline (1 - MAE_model / MAE_naive).
"""
from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, REPORTS, RETRAIN_EVERY, TEST_DAYS
from features import FEATURES, TARGET, load_features

LGBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    min_child_samples=40, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1,
    deterministic=True, force_col_wise=True, verbose=-1,
)


def make_models() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "lgbm": LGBMRegressor(**LGBM_PARAMS),
    }


def walk_forward(
    feats: pd.DataFrame,
    feature_cols: Sequence[str] = FEATURES,
    test_days_count: int = TEST_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """Run the declared expanding-window experiment.

    The split unit is a local delivery day.  All 23/24/25 observations from a
    delivery day stay together, and every training row strictly precedes the
    delivery day being forecast.
    """
    feature_cols = list(feature_cols)
    local_days = feats.index.tz_convert("Europe/Berlin").normalize()
    days = local_days.unique().sort_values()
    test_days = days[-test_days_count:]

    preds = {name: [] for name in ("naive_w", "ridge", "lgbm")}
    idx_all, y_all = [], []
    fitted = {}
    importance_sum = pd.Series(0.0, index=feature_cols)
    fit_count = 0

    for i, day in enumerate(test_days):
        if i % RETRAIN_EVERY == 0:
            train_mask = local_days < day
            X_tr = feats.loc[train_mask, feature_cols]
            y_tr = feats.loc[train_mask, TARGET]
            fitted = {n: m.fit(X_tr, y_tr) for n, m in make_models().items()}
            importance_sum += pd.Series(
                fitted["lgbm"].feature_importances_, index=feature_cols
            )
            fit_count += 1

        day_mask = local_days == day
        X_te = feats.loc[day_mask, feature_cols]
        y_te = feats.loc[day_mask, TARGET]
        idx_all.extend(X_te.index)
        y_all.extend(y_te)
        preds["naive_w"].extend(X_te["price_lag168"])
        for n in ("ridge", "lgbm"):
            preds[n].extend(fitted[n].predict(X_te))

    res = pd.DataFrame({"y_true": y_all, **preds},
                       index=pd.DatetimeIndex(idx_all, name="ts_utc"))

    metrics = {}
    mae_naive = mean_absolute_error(res["y_true"], res["naive_w"])
    for n in ("naive_w", "ridge", "lgbm"):
        mae = mean_absolute_error(res["y_true"], res[n])
        metrics[n] = {
            "MAE": round(mae, 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(res["y_true"], res[n]))), 2),
            "R2": round(r2_score(res["y_true"], res[n]), 3),
            "bias": round(float((res[n] - res["y_true"]).mean()), 2),
            "skill_vs_naive": round(1 - mae / mae_naive, 3),
        }
    importances = (importance_sum / max(fit_count, 1)).sort_values(ascending=False)
    return res, {
        "metrics": metrics,
        "importances": importances.to_dict(),
        "test_days": int(len(test_days)),
        "observations": int(len(res)),
        "refits": fit_count,
        "retrain_every_days": RETRAIN_EVERY,
    }


def main() -> None:
    feats = load_features()
    print(f"Feature matrix: {feats.shape[0]} rows, {len(FEATURES)} features")
    res, info = walk_forward(feats)

    res.to_csv(DATA / "predictions_oos.csv")
    (REPORTS / "model_metrics.json").write_text(json.dumps(info, indent=2))

    # submission file: out-of-sample LightGBM predictions
    sub = res[["lgbm"]].reset_index()
    sub.columns = ["id", "y_pred"]
    sub["y_pred"] = sub["y_pred"].round(2)
    sub.to_csv(DATA.parent / "submission.csv", index=False)

    print(json.dumps(info["metrics"], indent=2))
    print("Top features:",
          list(info["importances"].keys())[:8])


if __name__ == "__main__":
    main()
