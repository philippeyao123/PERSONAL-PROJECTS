from pathlib import Path

from systematic_research.config import ExperimentConfig
from systematic_research.tracking import set_global_seed, stable_hash


def test_configuration_round_trip_and_hash(tmp_path: Path) -> None:
    config = ExperimentConfig(name="test", seed=7)
    destination = tmp_path / "config.yaml"
    config.to_yaml(destination)
    loaded = ExperimentConfig.from_yaml(destination)
    assert loaded == config
    assert loaded.experiment_id == config.experiment_id
    assert stable_hash(config) == stable_hash(loaded)


def test_seed_reproduces_numpy_draws() -> None:
    first = set_global_seed(11).normal(size=5)
    second = set_global_seed(11).normal(size=5)
    assert (first == second).all()
