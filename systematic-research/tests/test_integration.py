from pathlib import Path

from systematic_research.config import ExperimentConfig, ValidationConfig
from systematic_research.examples.flagship import run_flagship


def test_flagship_pipeline_is_reproducible(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="integration",
        seed=123,
        validation=ValidationConfig(
            train_periods=504,
            validation_periods=100,
            test_periods=100,
            step_periods=100,
            purge_periods=2,
            embargo_periods=2,
        ),
    )
    first = run_flagship(config, tmp_path / "first")
    second = run_flagship(config, tmp_path / "second")
    assert first.markdown.exists()
    assert first.equity_chart.exists()
    assert (first.directory / "capacity.csv").exists()
    assert first.metrics_json.read_text() == second.metrics_json.read_text()
