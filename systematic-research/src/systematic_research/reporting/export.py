"""Stable CSV, JSON and optional Parquet exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

import pandas as pd

from systematic_research.exceptions import ResearchError


def export_frame(frame: pd.DataFrame, path: Union[str, Path]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(destination, index=False)
    elif suffix == ".json":
        frame.to_json(destination, orient="records", date_format="iso", indent=2)
    elif suffix == ".parquet":
        try:
            frame.to_parquet(destination, index=False)
        except ImportError as error:
            raise ResearchError(
                "Parquet export requires the optional dependency: pip install .[parquet]"
            ) from error
    else:
        raise ResearchError("export suffix must be .csv, .json or .parquet")
    return destination


def export_mapping(values: Mapping[str, Any], path: Union[str, Path]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(values), handle, indent=2, sort_keys=True, default=str)
    return destination
