from pathlib import Path

import pandas as pd

from systematic_research.cache import DeterministicCache
from systematic_research.reporting.export import export_frame, export_mapping


def test_cache_invalidates_with_data_hash(tmp_path: Path) -> None:
    cache = DeterministicCache(tmp_path / "cache")
    first_key = cache.key("feature", {"lookback": 20}, "data-a")
    second_key = cache.key("feature", {"lookback": 20}, "data-b")
    assert first_key != second_key
    cache.put(first_key, {"value": 1})
    assert cache.get(first_key) == {"value": 1}
    assert cache.get(second_key) is None


def test_csv_and_json_exports(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "value": [1.0]})
    assert export_frame(frame, tmp_path / "frame.csv").exists()
    assert export_frame(frame, tmp_path / "frame.json").exists()
    assert export_mapping({"metric": 1.0}, tmp_path / "metrics.json").exists()
