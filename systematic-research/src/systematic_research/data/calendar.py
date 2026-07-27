"""Safe calendar alignment."""

from __future__ import annotations

import pandas as pd

from systematic_research.exceptions import DataValidationError


def align_to_calendar(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    allow_price_fill: bool = False,
) -> pd.DataFrame:
    """Reindex each asset to a calendar; price forward-fill is rejected by default."""
    if not {"date", "asset"}.issubset(frame.columns):
        raise DataValidationError("calendar alignment requires date and asset")
    if allow_price_fill:
        raise DataValidationError(
            "price forward-fill is intentionally unsupported; model stale prices explicitly"
        )
    assets = sorted(frame["asset"].unique())
    index = pd.MultiIndex.from_product([calendar, assets], names=["date", "asset"])
    return frame.set_index(["date", "asset"]).reindex(index).reset_index()
