"""Cross-sectional exposure neutralization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def neutralize(
    frame: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    score_column: str = "score",
) -> pd.DataFrame:
    """Regress scores on declared exposures per date and return residual scores."""
    required = {"date", score_column, *exposure_columns}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"neutralization requires {sorted(required)}")
    result = frame.copy()

    def residualize(group: pd.DataFrame) -> pd.DataFrame:
        valid = group[[score_column, *exposure_columns]].dropna()
        residual = pd.Series(np.nan, index=group.index, dtype=float)
        if len(valid) > len(exposure_columns):
            design = np.column_stack(
                [np.ones(len(valid)), valid[list(exposure_columns)].to_numpy(dtype=float)]
            )
            target = valid[score_column].to_numpy(dtype=float)
            coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            residual.loc[valid.index] = target - design @ coefficients
        group = group.copy()
        group["neutral_score"] = residual
        return group

    groups = [residualize(group) for _, group in result.groupby("date", sort=False)]
    return pd.concat(groups, ignore_index=True)


def residual_exposure(
    frame: pd.DataFrame,
    exposure_column: str,
    *,
    weight_column: str = "target_weight",
) -> pd.Series:
    """Measure weighted portfolio exposure at every date."""
    required = {"date", exposure_column, weight_column}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"residual exposure requires {sorted(required)}")
    contribution = frame[exposure_column] * frame[weight_column]
    return contribution.groupby(frame["date"]).sum()
