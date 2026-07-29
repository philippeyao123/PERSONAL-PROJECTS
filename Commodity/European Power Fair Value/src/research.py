"""Publication-grade diagnostics for the European power fair-value study.

This module does not download data.  It consumes the frozen dataset and the
strictly walk-forward predictions produced by ``models.py`` and creates the
statistical evidence used by the paper:

* seven-day moving-block bootstrap confidence intervals;
* HLN-corrected forecast-loss tests on daily absolute-loss differentials;
* strictly prequential conformal intervals;
* feature-family ablations under the same walk-forward protocol;
* seasonal, intraday, renewable, and price-tail diagnostics;
* prompt-proxy signal sensitivity with right-censoring handled explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from config import DATA, REPORTS, RETRAIN_EVERY, TEST_DAYS
from features import (
    ALL_FEATURES,
    CALENDAR_FEATURES,
    FEATURES,
    RENEWABLE_FEATURES,
    TARGET,
    WEATHER_FEATURES,
    load_features,
)
from models import LGBM_PARAMS

SEED = 42
BOOTSTRAP_REPS = 4_000
MODELS = ("naive_w", "ridge", "ridge_hourly", "lgbm")
DEPENDENCE_BLOCK_DAYS = 7

ABLATIONS = {
    "primary": tuple(FEATURES),
    "weather_augmented": tuple(ALL_FEATURES),
    "fundamentals_calendar": RENEWABLE_FEATURES + CALENDAR_FEATURES,
    "no_renewables": tuple(c for c in FEATURES if c not in RENEWABLE_FEATURES),
    "calendar_only": CALENDAR_FEATURES,
}


def local_delivery_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return timezone-aware local delivery-day labels."""
    return index.tz_convert("Europe/Berlin").normalize()


def day_loss_table(res: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly errors by delivery day to respect intraday dependence."""
    out = pd.DataFrame(index=res.index)
    out["delivery_day"] = local_delivery_days(res.index)
    for model in MODELS:
        err = res[model] - res["y_true"]
        out[f"{model}_ae"] = err.abs()
        out[f"{model}_se"] = err.pow(2)
    return out.groupby("delivery_day").mean(numeric_only=True)


def moving_block_day_bootstrap_ci(
    values: np.ndarray,
    block: int = DEPENDENCE_BLOCK_DAYS,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile CI from overlapping blocks of delivery-day summaries."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    block = min(max(int(block), 1), len(x))
    starts = np.arange(len(x) - block + 1)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for rep in range(reps):
        indices = []
        while len(indices) < len(x):
            start = int(rng.choice(starts))
            indices.extend(range(start, start + block))
        draws[rep] = x[np.asarray(indices[: len(x)])].mean()
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def hac_mean_test(
    differential: np.ndarray,
    lag: int = DEPENDENCE_BLOCK_DAYS,
    forecast_horizon: int = 1,
) -> dict:
    """HLN-corrected two-sided test with a Bartlett Newey-West variance.

    ``differential`` is comparator loss minus candidate loss.  Positive
    values therefore favour the candidate model. The Harvey--Leybourne--Newbold
    finite-sample factor is reported explicitly; with one-day forecasts it is
    close to one but keeps the reference distribution honest.
    """
    d = np.asarray(differential, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    mean = float(d.mean())
    centered = d - mean
    lag = min(lag, max(n - 1, 0))
    long_run = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = math.sqrt(max(long_run, 0.0) / n) if n else float("nan")
    statistic = mean / se if se > 0 else float("nan")
    horizon = min(max(int(forecast_horizon), 1), max(n, 1))
    hln_scale = math.sqrt(
        max(n + 1 - 2 * horizon + horizon * (horizon - 1) / max(n, 1), 0.0)
        / max(n, 1)
    )
    hln_statistic = statistic * hln_scale
    hln_p_value = (
        float(2.0 * stats.t.sf(abs(hln_statistic), df=max(n - 1, 1)))
        if np.isfinite(hln_statistic)
        else float("nan")
    )
    return {
        "n_days": n,
        "mean_loss_improvement": mean,
        "hac_lag": lag,
        "hac_standard_error": se,
        "dm_statistic": statistic,
        "hln_scale": hln_scale,
        "hln_statistic": hln_statistic,
        "statistic": hln_statistic,
        "p_value": hln_p_value,
    }


def model_comparison(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = day_loss_table(res)
    unique_days = local_delivery_days(res.index).unique().sort_values()
    post_warmup = local_delivery_days(res.index) >= unique_days[90]
    rows = []
    for model in MODELS:
        ae = (res[model] - res["y_true"]).abs()
        se = (res[model] - res["y_true"]).pow(2)
        lo, hi = moving_block_day_bootstrap_ci(daily[f"{model}_ae"].to_numpy())
        post_error = (
            res.loc[post_warmup, model] - res.loc[post_warmup, "y_true"]
        ).abs()
        rows.append({
            "model": model,
            "observations": len(res),
            "delivery_days": len(daily),
            "mae": float(ae.mean()),
            "mae_ci_low": lo,
            "mae_ci_high": hi,
            "rmse": float(np.sqrt(se.mean())),
            "bias": float((res[model] - res["y_true"]).mean()),
            "median_ae": float(ae.median()),
            "q95_ae": float(ae.quantile(0.95)),
            "mae_after_first_90_days": float(post_error.mean()),
            "post_warmup_delivery_days": int(len(unique_days) - 90),
        })
    return pd.DataFrame(rows), daily


def dm_comparisons(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparator in ("naive_w", "ridge", "ridge_hourly"):
        test = hac_mean_test(
            daily[f"{comparator}_ae"].to_numpy()
            - daily["lgbm_ae"].to_numpy(),
            lag=7,
        )
        rows.append({"candidate": "lgbm", "comparator": comparator, **test})
    return pd.DataFrame(rows)


def prequential_conformal(
    res: pd.DataFrame,
    alpha: float = 0.10,
    calibration_days: int = 60,
    minimum_days: int = 30,
) -> pd.DataFrame:
    """Construct symmetric intervals using only residuals from earlier days."""
    days = local_delivery_days(res.index)
    unique_days = days.unique().sort_values()
    abs_error = (res["lgbm"] - res["y_true"]).abs()
    frames = []
    for i, day in enumerate(unique_days):
        if i < minimum_days:
            continue
        prior_days = unique_days[max(0, i - calibration_days):i]
        calibration = abs_error[days.isin(prior_days)].to_numpy()
        n_cal = len(calibration)
        probability = min(1.0, math.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        radius = float(np.quantile(calibration, probability, method="higher"))
        mask = days == day
        part = res.loc[mask, ["y_true", "lgbm"]].copy()
        part["delivery_day"] = day
        part["calibration_n"] = n_cal
        part["radius"] = radius
        part["lower"] = part["lgbm"] - radius
        part["upper"] = part["lgbm"] + radius
        part["covered"] = (
            (part["y_true"] >= part["lower"])
            & (part["y_true"] <= part["upper"])
        )
        lower_error = part["y_true"] - part["lower"]
        upper_error = part["y_true"] - part["upper"]
        part["lower_pinball"] = np.maximum(
            0.05 * lower_error, -0.95 * lower_error
        )
        part["upper_pinball"] = np.maximum(
            0.95 * upper_error, -0.05 * upper_error
        )
        part["interval_score"] = (
            part["upper"] - part["lower"]
            + (2.0 / alpha)
            * (part["lower"] - part["y_true"])
            * (part["y_true"] < part["lower"])
            + (2.0 / alpha)
            * (part["y_true"] - part["upper"])
            * (part["y_true"] > part["upper"])
        )
        part["hour"] = part.index.tz_convert("Europe/Berlin").hour
        frames.append(part)
    return pd.concat(frames).sort_index()


def conformal_hour_tests(
    conformal: pd.DataFrame,
    nominal: float = 0.90,
) -> pd.DataFrame:
    """Exact binomial hour-conditional coverage tests with Holm adjustment."""
    rows = []
    for hour, group in conformal.groupby("hour"):
        n = len(group)
        successes = int(group["covered"].sum())
        coverage = successes / n
        standard_error = math.sqrt(nominal * (1.0 - nominal) / n)
        z = (coverage - nominal) / standard_error
        p_value = float(
            stats.binomtest(successes, n, nominal, alternative="two-sided").pvalue
        )
        rows.append(
            {
                "hour": int(hour),
                "observations": n,
                "covered": successes,
                "coverage": coverage,
                "z_statistic": z,
                "p_value": p_value,
                "direction": "over" if coverage > nominal else "under",
            }
        )
    result = pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)
    order = np.argsort(result["p_value"].to_numpy())
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(
            result.loc[order, "p_value"].to_numpy()
            * (len(result) - np.arange(len(result))),
            1.0,
        )
    )
    adjusted = np.empty(len(result), dtype=float)
    adjusted[order] = adjusted_sorted
    result["holm_p_value"] = adjusted
    result["significant_normal_5pct"] = result["z_statistic"].abs() > 1.96
    result["significant_holm_5pct"] = result["holm_p_value"] < 0.05
    return result


def prequential_bias_correction(
    res: pd.DataFrame,
    window_days: int = 60,
    minimum_days: int = 30,
) -> pd.DataFrame:
    """Apply a rolling intercept correction estimated only from prior days."""
    days = local_delivery_days(res.index)
    unique_days = days.unique().sort_values()
    residual = res["lgbm"] - res["y_true"]
    frames = []
    for i, day in enumerate(unique_days):
        if i < minimum_days:
            continue
        prior_days = unique_days[max(0, i - window_days):i]
        estimated_bias = float(residual[days.isin(prior_days)].mean())
        part = res.loc[days == day, ["y_true", "lgbm"]].copy()
        part["estimated_bias"] = estimated_bias
        part["bias_corrected"] = part["lgbm"] - estimated_bias
        frames.append(part)
    return pd.concat(frames).sort_index()


def _lgbm_ablation_predictions(
    feats: pd.DataFrame,
    feature_cols: tuple[str, ...],
) -> pd.Series:
    local_days = local_delivery_days(feats.index)
    unique_days = local_days.unique().sort_values()
    test_days = unique_days[-TEST_DAYS:]
    values = []
    indices = []
    fitted = None
    for i, day in enumerate(test_days):
        if i % RETRAIN_EVERY == 0:
            train = local_days < day
            fitted = LGBMRegressor(**LGBM_PARAMS)
            fitted.fit(feats.loc[train, list(feature_cols)], feats.loc[train, TARGET])
        test = local_days == day
        values.extend(fitted.predict(feats.loc[test, list(feature_cols)]))
        indices.extend(feats.index[test])
    return pd.Series(values, index=pd.DatetimeIndex(indices), dtype=float)


def run_ablations(
    feats: pd.DataFrame,
    weather_feats: pd.DataFrame,
    res: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.DataFrame(index=res.index)
    predictions["y_true"] = res["y_true"]
    predictions["primary"] = res["lgbm"]
    for name, cols in ABLATIONS.items():
        if name == "primary":
            continue
        source = weather_feats if name == "weather_augmented" else feats
        pred = _lgbm_ablation_predictions(source, cols)
        predictions[name] = pred.reindex(predictions.index)

    rows = []
    primary_mae = float(
        (predictions["primary"] - predictions["y_true"]).abs().mean()
    )
    for name in ABLATIONS:
        err = predictions[name] - predictions["y_true"]
        by_day = err.abs().groupby(local_delivery_days(predictions.index)).mean()
        lo, hi = moving_block_day_bootstrap_ci(
            by_day.to_numpy(), seed=SEED + len(rows)
        )
        rows.append({
            "specification": name,
            "features": len(ABLATIONS[name]),
            "mae": float(err.abs().mean()),
            "mae_ci_low": lo,
            "mae_ci_high": hi,
            "rmse": float(np.sqrt(err.pow(2).mean())),
            "bias": float(err.mean()),
            "mae_change_vs_primary_pct": float(
                100.0 * (err.abs().mean() / primary_mae - 1.0)
            ),
        })
    return predictions, pd.DataFrame(rows).sort_values("mae")


def regime_metrics(res: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    frame = res.join(
        dataset[["fcst_solar", "fcst_wind_total"]],
        how="left",
    )
    local = frame.index.tz_convert("Europe/Berlin")
    month = local.month
    frame["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )
    hour = local.hour
    frame["hour_block"] = np.select(
        [hour <= 6, hour <= 15, hour <= 21],
        ["night", "daylight", "evening_ramp"],
        default="late_evening",
    )
    frame["price_regime"] = np.select(
        [frame["y_true"] < 0, frame["y_true"] >= 200],
        ["negative", "above_200"],
        default="regular",
    )
    renewables = frame["fcst_solar"] + frame["fcst_wind_total"]
    frame["renewable_quartile"] = pd.qcut(
        renewables, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
    ).astype(str)

    rows = []
    for dimension in ("season", "hour_block", "price_regime", "renewable_quartile"):
        for level, group in frame.groupby(dimension, observed=True):
            for model in MODELS:
                err = group[model] - group["y_true"]
                rows.append({
                    "dimension": dimension,
                    "level": str(level),
                    "model": model,
                    "observations": len(group),
                    "mae": float(err.abs().mean()),
                    "rmse": float(np.sqrt(err.pow(2).mean())),
                    "bias": float(err.mean()),
                })
    return pd.DataFrame(rows)


def tail_metrics(res: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "negative": res["y_true"] < 0,
        "regular": (res["y_true"] >= 0) & (res["y_true"] < 200),
        "above_200": res["y_true"] >= 200,
    }
    rows = []
    for regime, mask in masks.items():
        for model in MODELS:
            err = res.loc[mask, model] - res.loc[mask, "y_true"]
            rows.append({
                "regime": regime,
                "model": model,
                "observations": int(mask.sum()),
                "mae": float(err.abs().mean()),
                "bias": float(err.mean()),
            })
    negative = res["y_true"] < 0
    high = res["y_true"] >= 200
    for model in MODELS:
        rows.append({
            "regime": "negative_detection",
            "model": model,
            "observations": int(negative.sum()),
            "mae": float(((res.loc[negative, model] < 0).mean())),
            "bias": float("nan"),
        })
        rows.append({
            "regime": "above_200_detection",
            "model": model,
            "observations": int(high.sum()),
            "mae": float(((res.loc[high, model] >= 200).mean())),
            "bias": float("nan"),
        })
    return pd.DataFrame(rows)


def _signal_frame(
    res: pd.DataFrame,
    prompt_window: int,
    forecast_column: str = "lgbm",
) -> pd.DataFrame:
    local = local_delivery_days(res.index)
    daily = res.groupby(local).agg(
        fair_value=(forecast_column, "mean"),
        realised=("y_true", "mean"),
    )
    daily["prompt_proxy"] = (
        daily["realised"].shift(1).rolling(prompt_window).mean()
    )
    daily["gap"] = daily["fair_value"] - daily["prompt_proxy"]
    rolling = daily["gap"].rolling(60, min_periods=30)
    daily["gap_z"] = (daily["gap"] - rolling.mean()) / rolling.std()
    daily["forward_week"] = (
        daily["realised"][::-1].rolling(7, min_periods=7).mean()[::-1]
    )
    daily["forward_spread"] = daily["forward_week"] - daily["prompt_proxy"]
    return daily.dropna(subset=["gap_z"])


def moving_block_signal_ci(
    captured: np.ndarray,
    active: np.ndarray,
    block: int = 14,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> dict:
    """Moving-block bootstrap for overlapping seven-day signal outcomes."""
    values = np.asarray(captured, dtype=float)
    is_active = np.asarray(active, dtype=bool)
    n = len(values)
    starts = np.arange(max(n - block + 1, 1))
    rng = np.random.default_rng(seed)
    means, hits = [], []
    for _ in range(reps):
        idx = []
        while len(idx) < n:
            start = int(rng.choice(starts))
            idx.extend(range(start, min(start + block, n)))
        idx = np.asarray(idx[:n])
        mask = is_active[idx] & np.isfinite(values[idx])
        if mask.any():
            sample = values[idx][mask]
            means.append(float(sample.mean()))
            hits.append(float((sample > 0).mean()))
    return {
        "block_days": block,
        "mean_ci": np.quantile(means, [0.025, 0.975]).astype(float).tolist(),
        "hit_rate_ci": np.quantile(hits, [0.025, 0.975]).astype(float).tolist(),
    }


def signal_sensitivity(res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    default_arrays = None
    for prompt_window in (5, 7, 14):
        daily = _signal_frame(res, prompt_window)
        for threshold in (0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
            sign = np.sign(daily["gap_z"])
            active = daily["gap_z"].abs() > threshold
            captured = sign * daily["forward_spread"]
            evaluable = active & captured.notna()
            sample = captured[evaluable]
            rows.append({
                "prompt_window": prompt_window,
                "threshold": threshold,
                "signal_days": len(daily),
                "evaluable_active_days": int(evaluable.sum()),
                "right_censored_active_days": int((active & captured.isna()).sum()),
                "hit_rate": float((sample > 0).mean()),
                "average_captured": float(sample.mean()),
                "median_captured": float(sample.median()),
                "long_days": int((evaluable & (sign > 0)).sum()),
                "short_days": int((evaluable & (sign < 0)).sum()),
            })
            if prompt_window == 7 and threshold == 0.75:
                default_arrays = (
                    captured.to_numpy(),
                    active.to_numpy(),
                )
    default_ci = moving_block_signal_ci(*default_arrays)
    return pd.DataFrame(rows), default_ci


def signal_benchmarks(
    res: pd.DataFrame,
    prompt_window: int = 7,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Bound the prompt diagnostic with persistence and infeasible references."""
    local = local_delivery_days(res.index)
    realised = res["y_true"].groupby(local).mean()
    prompt = realised.shift(1).rolling(prompt_window).mean()
    forward = realised[::-1].rolling(7, min_periods=7).mean()[::-1]
    forward_spread = forward - prompt
    always = forward_spread.dropna()
    rows = [
        {
            "input": "always_long",
            "feasible": True,
            "evaluable_active_days": len(always),
            "hit_rate": float((always > 0).mean()),
            "average_captured": float(always.mean()),
            "interpretation": "absence-of-signal persistence reference",
        }
    ]
    for label, column, feasible in (
        ("weekly_naive", "naive_w", True),
        ("ridge", "ridge", True),
        ("ridge_hourly", "ridge_hourly", True),
        ("lightgbm", "lgbm", True),
        ("perfect_day_d", "y_true", False),
    ):
        daily = _signal_frame(res, prompt_window, column)
        sign = np.sign(daily["gap_z"])
        active = daily["gap_z"].abs() > threshold
        captured = sign * daily["forward_spread"]
        evaluable = active & captured.notna()
        sample = captured[evaluable]
        rows.append(
            {
                "input": label,
                "feasible": feasible,
                "evaluable_active_days": int(evaluable.sum()),
                "hit_rate": float((sample > 0).mean()),
                "average_captured": float(sample.mean()),
                "interpretation": (
                    "infeasible same-day perfect-information upper bound"
                    if not feasible
                    else "same signal construction"
                ),
            }
        )
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    res = pd.read_csv(DATA / "predictions_oos.csv", index_col=0)
    res.index = pd.to_datetime(res.index, utc=True)
    dataset = pd.read_csv(DATA / "dataset.csv", index_col=0)
    dataset.index = pd.to_datetime(dataset.index, utc=True)
    feats = load_features(FEATURES)
    weather_feats = load_features(ALL_FEATURES)

    comparison, daily_losses = model_comparison(res)
    dm = dm_comparisons(daily_losses)
    conformal = prequential_conformal(res)
    coverage_tests = conformal_hour_tests(conformal)
    bias_corrected = prequential_bias_correction(res)
    ablation_predictions, ablations = run_ablations(feats, weather_feats, res)
    regimes = regime_metrics(res, dataset)
    tails = tail_metrics(res)
    signals, signal_ci = signal_sensitivity(res)
    signal_bounds = signal_benchmarks(res)

    comparison.to_csv(DATA / "model_comparison.csv", index=False)
    daily_losses.to_csv(DATA / "model_daily_losses.csv")
    dm.to_csv(DATA / "dm_tests.csv", index=False)
    conformal.to_csv(DATA / "conformal_diagnostics.csv")
    coverage_tests.to_csv(DATA / "conformal_hour_tests.csv", index=False)
    bias_corrected.to_csv(DATA / "bias_correction_diagnostics.csv")
    ablation_predictions.to_csv(DATA / "ablation_predictions.csv")
    ablations.to_csv(DATA / "ablation_metrics.csv", index=False)
    regimes.to_csv(DATA / "regime_metrics.csv", index=False)
    tails.to_csv(DATA / "tail_metrics.csv", index=False)
    signals.to_csv(DATA / "signal_sensitivity.csv", index=False)
    signal_bounds.to_csv(DATA / "signal_benchmarks.csv", index=False)

    coverage = float(conformal["covered"].mean())
    coverage_by_hour = (
        conformal.groupby("hour")["covered"].mean().astype(float).to_dict()
    )
    corrected_error = bias_corrected["bias_corrected"] - bias_corrected["y_true"]
    uncorrected_error = bias_corrected["lgbm"] - bias_corrected["y_true"]
    primary_days = local_delivery_days(feats.index).unique().sort_values()
    metrics = {
        "data": {
            "rows": len(dataset),
            "start_utc": str(dataset.index.min()),
            "end_utc": str(dataset.index.max()),
            "dataset_sha256": sha256(DATA / "dataset.csv"),
            "predictions_sha256": sha256(DATA / "predictions_oos.csv"),
            "primary_feature_rows": len(feats),
            "weather_feature_rows": len(weather_feats),
            "pre_oos_training_days": int(len(primary_days) - len(local_delivery_days(res.index).unique())),
        },
        "model_comparison": comparison.to_dict(orient="records"),
        "dm_tests": dm.to_dict(orient="records"),
        "conformal": {
            "nominal_coverage": 0.90,
            "empirical_coverage": coverage,
            "observations": len(conformal),
            "mean_width": float((conformal["upper"] - conformal["lower"]).mean()),
            "mean_interval_score": float(conformal["interval_score"].mean()),
            "mean_lower_pinball": float(conformal["lower_pinball"].mean()),
            "mean_upper_pinball": float(conformal["upper_pinball"].mean()),
            "coverage_by_hour": coverage_by_hour,
            "normal_approx_significant_hours": int(
                coverage_tests["significant_normal_5pct"].sum()
            ),
            "holm_significant_hours": int(
                coverage_tests["significant_holm_5pct"].sum()
            ),
            "undercover_significant_hours": int(
                (
                    coverage_tests["significant_normal_5pct"]
                    & (coverage_tests["direction"] == "under")
                ).sum()
            ),
            "overcover_significant_hours": int(
                (
                    coverage_tests["significant_normal_5pct"]
                    & (coverage_tests["direction"] == "over")
                ).sum()
            ),
            "calibration_days": 60,
            "minimum_days": 30,
        },
        "bias_correction": {
            "observations": len(bias_corrected),
            "uncorrected_mae": float(uncorrected_error.abs().mean()),
            "corrected_mae": float(corrected_error.abs().mean()),
            "uncorrected_bias": float(uncorrected_error.mean()),
            "corrected_bias": float(corrected_error.mean()),
        },
        "ablations": ablations.to_dict(orient="records"),
        "default_signal_block_bootstrap": signal_ci,
        "signal_benchmarks": signal_bounds.to_dict(orient="records"),
    }
    (REPORTS / "research_metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2)
    )
    print(json.dumps(_jsonable(metrics), indent=2))


if __name__ == "__main__":
    main()
