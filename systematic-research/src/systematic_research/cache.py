"""Content-addressed cache for deterministic pipeline stages only."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from systematic_research.tracking import stable_hash

T = TypeVar("T")


class DeterministicCache:
    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, stage: str, config: Any, data_hash: str) -> str:
        return stable_hash({"stage": stage, "config": config, "data_hash": data_hash})

    def get(self, key: str) -> Optional[Any]:
        path = self.directory / f"{key}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def put(self, key: str, value: Any) -> Path:
        path = self.directory / f"{key}.pkl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return path

    def get_or_compute(self, key: str, compute: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.put(key, value)
        return value
