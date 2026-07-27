"""In-sample pair selection and OU diagnostics example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


@dataclass(frozen=True)
class PairModel:
    asset_x: str
    asset_y: str
    hedge_ratio: float
    mean: float
    half_life: float


def fit_pair_in_sample(
    prices: pd.DataFrame,
    asset_x: str,
    asset_y: str,
    train_end: pd.Timestamp,
) -> PairModel:
    """Fit the pair only with observations at or before train_end."""
    sample = prices.loc[prices.index <= train_end, [asset_x, asset_y]].dropna()
    if len(sample) < 30:
        raise DataValidationError("pair fitting requires at least 30 in-sample observations")
    design = np.column_stack([np.ones(len(sample)), sample[asset_y].to_numpy()])
    coefficients, _, _, _ = np.linalg.lstsq(design, sample[asset_x].to_numpy(), rcond=None)
    spread = sample[asset_x] - coefficients[1] * sample[asset_y]
    lagged = spread.shift(1).dropna()
    changes = spread.diff().dropna().reindex(lagged.index)
    ar_design = np.column_stack([np.ones(len(lagged)), lagged.to_numpy()])
    ar_coefficients, _, _, _ = np.linalg.lstsq(ar_design, changes.to_numpy(), rcond=None)
    speed = max(-float(ar_coefficients[1]), 1e-9)
    return PairModel(
        asset_x,
        asset_y,
        float(coefficients[1]),
        float(spread.mean()),
        float(np.log(2.0) / speed),
    )
