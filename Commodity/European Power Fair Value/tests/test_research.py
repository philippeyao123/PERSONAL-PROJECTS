from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import ALL_FEATURES, FEATURES, build_features, load_features  # noqa: E402
from qa import run_qa  # noqa: E402
from research import (  # noqa: E402
    conformal_hour_tests,
    hac_mean_test,
    moving_block_day_bootstrap_ci,
    prequential_conformal,
)
from trading import build_views  # noqa: E402


def test_price_lags_use_only_earlier_observations() -> None:
    index = pd.date_range("2026-01-01", periods=220, freq="h", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["price"] = np.arange(len(index), dtype=float)
    frame["fcst_solar"] = 1.0
    frame["fcst_wind_total"] = 2.0
    frame["wx_temperature_2m"] = 10.0
    frame["wx_wind_speed_100m"] = 5.0
    frame["wx_shortwave_radiation"] = 0.0
    features = build_features(frame)
    row = features.iloc[-1]
    position = len(frame) - 1
    assert row["price_lag24"] == frame["price"].iloc[position - 24]
    assert row["price_lag168"] == frame["price"].iloc[position - 168]


def test_frozen_dataset_passes_structural_qa() -> None:
    frame = pd.read_csv(ROOT / "data/dataset.csv", index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    report = run_qa(frame)
    assert report["status"] == "PASS"
    assert report["duplicates"] == 0
    assert report["completeness"]["missing_hours"] == 0


def test_hac_test_sign_favours_lower_candidate_loss() -> None:
    differential = np.linspace(1.0, 2.0, 100)
    result = hac_mean_test(differential, lag=7)
    assert result["mean_loss_improvement"] > 0
    assert result["hln_statistic"] > 0
    assert 0 < result["hln_scale"] <= 1
    assert result["p_value"] < 0.01


def test_primary_sample_is_not_conditioned_on_weather() -> None:
    primary = load_features(FEATURES)
    weather = load_features(ALL_FEATURES)
    assert len(primary) >= len(weather)
    assert primary[FEATURES + ["price"]].notna().all().all()


def test_moving_block_bootstrap_is_seeded_and_order_aware() -> None:
    values = np.arange(30, dtype=float)
    first = moving_block_day_bootstrap_ci(values, block=7, reps=200, seed=8)
    second = moving_block_day_bootstrap_ci(values, block=7, reps=200, seed=8)
    assert first == second
    assert first[0] < values.mean() < first[1]


def test_conformal_calibration_is_strictly_prequential() -> None:
    index = pd.date_range("2025-01-01", periods=90 * 24, freq="h", tz="UTC")
    truth = np.sin(np.arange(len(index)) / 24)
    frame = pd.DataFrame(
        {"y_true": truth, "lgbm": truth + 0.1},
        index=index,
    )
    original = prequential_conformal(frame, calibration_days=30, minimum_days=10)
    perturbed = frame.copy()
    perturbed.loc[index[-24:], "y_true"] += 1000
    changed = prequential_conformal(perturbed, calibration_days=30, minimum_days=10)
    cutoff = index[-24]
    pd.testing.assert_series_equal(
        original.loc[original.index < cutoff, "radius"],
        changed.loc[changed.index < cutoff, "radius"],
    )


def test_walk_forward_artifact_is_one_complete_year() -> None:
    predictions = pd.read_csv(ROOT / "data/predictions_oos.csv", index_col=0)
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    days = predictions.index.tz_convert("Europe/Berlin").normalize()
    assert len(days.unique()) == 365
    assert len(predictions) == 8760
    assert not predictions.index.duplicated().any()
    assert {"naive_w", "ridge", "ridge_hourly", "lgbm"}.issubset(predictions)


def test_trading_statistics_exclude_right_censored_views() -> None:
    views = build_views()
    assert (~views["is_evaluable"]).sum() == 6
    assert views.loc[~views["is_evaluable"], "captured"].isna().all()


def test_ablation_full_column_matches_frozen_lightgbm() -> None:
    predictions = pd.read_csv(ROOT / "data/predictions_oos.csv", index_col=0)
    ablations = pd.read_csv(ROOT / "data/ablation_predictions.csv", index_col=0)
    np.testing.assert_allclose(
        predictions["lgbm"], ablations["primary"], rtol=0, atol=1e-12
    )


def test_reported_conformal_coverage_matches_rows() -> None:
    diagnostics = pd.read_csv(ROOT / "data/conformal_diagnostics.csv")
    metrics = json.loads((ROOT / "reports/research_metrics.json").read_text())
    assert abs(
        diagnostics["covered"].mean()
        - metrics["conformal"]["empirical_coverage"]
    ) < 1e-12
    tests = conformal_hour_tests(diagnostics)
    assert tests["significant_normal_5pct"].sum() == metrics["conformal"][
        "normal_approx_significant_hours"
    ]


def test_signal_benchmarks_include_naive_and_perfect_information_bounds() -> None:
    frame = pd.read_csv(ROOT / "data/signal_benchmarks.csv").set_index("input")
    assert {"always_long", "weekly_naive", "ridge_hourly", "lightgbm", "perfect_day_d"}.issubset(
        frame.index
    )
    assert not bool(frame.loc["perfect_day_d", "feasible"])


def test_dataset_fingerprint_matches_report() -> None:
    digest = hashlib.sha256((ROOT / "data/dataset.csv").read_bytes()).hexdigest()
    metrics = json.loads((ROOT / "reports/research_metrics.json").read_text())
    assert digest == metrics["data"]["dataset_sha256"]


def test_all_publication_figures_exist_as_vector_pdf() -> None:
    for number in range(1, 15):
        matches = list((ROOT / "figures").glob(f"{number:02d}_*.pdf"))
        assert len(matches) == 1
        assert matches[0].stat().st_size > 1_000
