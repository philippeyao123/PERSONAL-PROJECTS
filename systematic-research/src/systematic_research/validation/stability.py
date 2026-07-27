"""Parameter, sub-period, regime and placebo diagnostics."""

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def parameter_sensitivity(
    parameter_grid: Mapping[str, Sequence[float]],
    evaluator: Callable[[Dict[str, float]], float],
) -> pd.DataFrame:
    if not parameter_grid:
        raise DataValidationError("parameter grid cannot be empty")
    names = list(parameter_grid)
    rows = []
    for values in product(*(parameter_grid[name] for name in names)):
        parameters = dict(zip(names, values))
        rows.append({**parameters, "score": evaluator(parameters)})
    return pd.DataFrame(rows)


def subperiod_performance(
    returns: pd.Series,
    periods: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
    evaluator: Callable[[pd.Series], float],
) -> pd.DataFrame:
    rows = []
    for name, (start, end) in periods.items():
        sample = returns.loc[(returns.index >= start) & (returns.index <= end)]
        rows.append({"period": name, "score": evaluator(sample), "observations": len(sample)})
    return pd.DataFrame(rows)


def delayed_signal_placebo(signal: pd.Series, periods: int) -> pd.Series:
    if periods < 1:
        raise DataValidationError("placebo delay must be positive")
    return signal.shift(periods)


def randomized_signal_placebo(signal: pd.Series, seed: int) -> pd.Series:
    generator = np.random.default_rng(seed)
    values = signal.to_numpy(copy=True)
    generator.shuffle(values)
    return pd.Series(values, index=signal.index, name=signal.name)


def permuted_labels(labels: pd.Series, seed: int) -> pd.Series:
    return randomized_signal_placebo(labels, seed)
