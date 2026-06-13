"""Factor library.

Every factor is a class with a common signature:

    compute(panel, as_of) -> pd.Series   # cross-sectional scores at `as_of`

This uniformity lets the engine treat factors interchangeably and lets the
combiner stack them without special-casing. All factors return RAW scores;
neutralization / z-scoring / winsorization happen downstream in the combiner
so that the transformation pipeline is consistent and testable in one place.
"""
from __future__ import annotations

import abc

import pandas as pd

from alpha_factory.data.loader import PanelData


class Factor(abc.ABC):
    """Base class for all factors.

    Subclasses implement `_raw_scores`. The public `compute` wraps it and
    guarantees the output is a Series indexed by asset, restricted to names
    that are tradable (non-NaN price) at `as_of`.
    """

    #: Human-readable name used in reports and column headers.
    name: str = "factor"
    #: Sign convention: +1 means "higher score = expected higher return".
    direction: int = 1

    @abc.abstractmethod
    def _raw_scores(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        ...

    def compute(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        scores = self._raw_scores(panel, as_of)
        # Keep only currently tradable names.
        tradable = panel.prices.loc[:as_of].iloc[-1].dropna().index
        scores = scores.reindex(tradable)
        return self.direction * scores.dropna()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Factor {self.name} dir={self.direction:+d}>"


class Momentum(Factor):
    """Cross-sectional price momentum, skipping the most recent month.

    Classic 12-1: total return from t-`lookback` to t-`skip`, the skip avoiding
    short-term reversal contamination.
    """

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip
        self.name = f"momentum_{lookback}_{skip}"

    def _raw_scores(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        px = panel.prices.loc[:as_of]
        if len(px) < self.lookback + 1:
            return pd.Series(dtype=float)
        end = px.iloc[-1 - self.skip]
        start = px.iloc[-1 - self.lookback]
        return (end / start - 1.0)


class ShortTermReversal(Factor):
    """Short-horizon reversal: recent winners tend to underperform.

    direction = -1 because high recent return => expected lower forward return.
    """

    direction = -1

    def __init__(self, lookback: int = 21) -> None:
        self.lookback = lookback
        self.name = f"reversal_{lookback}"

    def _raw_scores(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        px = panel.prices.loc[:as_of]
        if len(px) < self.lookback + 1:
            return pd.Series(dtype=float)
        return px.iloc[-1] / px.iloc[-1 - self.lookback] - 1.0


class LowVolatility(Factor):
    """Low-volatility anomaly: lower realized vol => higher risk-adjusted return.

    direction = -1 so that low-vol names score high.
    """

    direction = -1

    def __init__(self, lookback: int = 126) -> None:
        self.lookback = lookback
        self.name = f"lowvol_{lookback}"

    def _raw_scores(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        rets = panel.returns.loc[:as_of].tail(self.lookback)
        if len(rets) < self.lookback // 2:
            return pd.Series(dtype=float)
        return rets.std()


class FundamentalValue(Factor):
    """Value factor from a PIT-lagged fundamental field (e.g. earnings yield)."""

    def __init__(self, field: str = "earnings_yield", direction: int = 1) -> None:
        self.field = field
        self.direction = direction
        self.name = f"value_{field}"

    def _raw_scores(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        if self.field not in panel.fundamentals:
            raise KeyError(f"fundamental '{self.field}' not in panel")
        frame = panel.fundamentals[self.field].loc[:as_of]
        if frame.empty:
            return pd.Series(dtype=float)
        return frame.iloc[-1]


class Amihud(Factor):
    """Amihud illiquidity proxy: |return| / dollar-volume.

    Without volume data we approximate with |return| magnitude (a crude
    illiquidity proxy). direction = +1 captures the illiquidity premium.
    Kept deliberately simple; swap in true ADV when volume is available.
    """

    def __init__(self, lookback: int = 63) -> None:
        self.lookback = lookback
        self.name = f"amihud_{lookback}"

    def _raw_scores(self, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
        rets = panel.returns.loc[:as_of].tail(self.lookback)
        if len(rets) < self.lookback // 2:
            return pd.Series(dtype=float)
        return rets.abs().mean()
