from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from systematic_research import cli
from systematic_research.benchmarks import (
    cash_benchmark,
    equal_weight_benchmark,
    market_benchmark,
)
from systematic_research.config import ExperimentConfig
from systematic_research.cpp_bridge import load_qf_rates
from systematic_research.exceptions import DataValidationError, ResearchError
from systematic_research.logging import JsonFormatter
from systematic_research.signals.neutralization import neutralize, residual_exposure
from systematic_research.validation.stability import (
    delayed_signal_placebo,
    parameter_sensitivity,
    permuted_labels,
    randomized_signal_placebo,
    subperiod_performance,
)
from systematic_research.validation.walk_forward import assert_no_label_overlap


def test_neutralization_removes_linear_exposure() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 6 + ["2024-01-03"] * 6),
            "asset": list("ABCDEF") * 2,
            "score": [0.1, 2.2, 3.8, 6.1, 8.2, 9.9] * 2,
            "beta": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        }
    )

    result = neutralize(frame, exposure_columns=["beta"])
    assert "neutral_score" in result
    for _, group in result.groupby("date"):
        assert abs(float(group["neutral_score"].mean())) < 1e-10

    result["target_weight"] = result["neutral_score"]
    assert residual_exposure(result, "beta").abs().max() < 1e-10


def test_neutralization_validates_columns() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "asset": ["A"], "score": [1.0]})
    with pytest.raises(DataValidationError):
        neutralize(frame, exposure_columns=["missing"])


def test_stability_and_placebo_helpers_are_deterministic() -> None:
    grid = {"lookback": [20, 60], "threshold": [0.0, 0.5]}
    sensitivity = parameter_sensitivity(
        grid, lambda params: params["lookback"] / 100 + params["threshold"]
    )
    assert len(sensitivity) == 4
    assert "score" in sensitivity

    dates = pd.date_range("2021-01-01", periods=12, freq="D")
    periods = {
        "first": (dates[0], dates[3]),
        "middle": (dates[4], dates[7]),
        "last": (dates[8], dates[11]),
    }
    performance = subperiod_performance(
        pd.Series(np.arange(12, dtype=float), index=dates),
        periods=periods,
        evaluator=lambda series: float(series.mean()),
    )
    assert len(performance) == 3

    signal = pd.Series(np.arange(12, dtype=float), name="score")
    assert delayed_signal_placebo(signal, periods=2).isna().sum() == 2
    random_a = randomized_signal_placebo(signal, seed=7)
    random_b = randomized_signal_placebo(signal, seed=7)
    pd.testing.assert_series_equal(random_a, random_b)
    assert sorted(permuted_labels(signal, seed=11).tolist()) == sorted(signal.tolist())


def test_benchmarks_cover_cash_market_and_equal_weight() -> None:
    returns = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2024-01-01", periods=3), 2),
            "asset": ["A", "B"] * 3,
            "return": [0.01, 0.03, -0.02, 0.02, 0.01, -0.01],
        }
    )
    assert cash_benchmark(pd.Index(range(3))).eq(0.0).all()
    assert equal_weight_benchmark(returns).iloc[0] == pytest.approx(0.02)
    assert market_benchmark(returns, "A").iloc[0] == pytest.approx(0.01)
    with pytest.raises(DataValidationError):
        market_benchmark(returns, "ABSENT")


def test_label_overlap_detection() -> None:
    train = pd.DataFrame(
        {
            "label_start": pd.date_range("2020-01-01", periods=5),
            "label_end": pd.date_range("2020-01-02", periods=5),
        }
    )
    safe_test = pd.DatetimeIndex(pd.date_range("2020-01-07", periods=3))
    assert_no_label_overlap(train, safe_test)

    overlapping_test = pd.DatetimeIndex(pd.date_range("2020-01-05", periods=3))
    with pytest.raises(DataValidationError):
        assert_no_label_overlap(train, overlapping_test)


def test_json_formatter_and_optional_cpp_bridge() -> None:
    record = logging.LogRecord(
        name="systematic_research",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="completed",
        args=(),
        exc_info=None,
    )
    record.context = {"run_id": "unit-test"}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "completed"
    assert payload["context"]["run_id"] == "unit-test"

    with pytest.raises(ResearchError, match="optional"):
        load_qf_rates()


def test_cli_dispatches_flagship(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    markdown = tmp_path / "report.md"
    markdown.write_text("# report\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(config: ExperimentConfig, output_dir: Path) -> SimpleNamespace:
        captured["name"] = config.name
        captured["output"] = output_dir
        return SimpleNamespace(markdown=markdown)

    monkeypatch.setattr(cli, "run_flagship", fake_run)
    config = tmp_path / "config.yaml"
    config.write_text("name: cli-test\nseed: 5\nperiods_per_year: 252\n", encoding="utf-8")
    output = tmp_path / "out"

    assert cli.main(["run", "--config", str(config), "--output", str(output)]) == 0
    assert captured == {"name": "cli-test", "output": output}
