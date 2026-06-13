"""Data ingestion layer with point-in-time (PIT) discipline.

The single most important property of this module: NO LOOK-AHEAD BIAS.
- Fundamentals are lagged to reflect realistic reporting delays.
- Delisted names are retained (no survivorship bias).
- All panels are indexed (date, asset) and aligned to a common trading calendar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PanelData:
    """A multi-asset panel with strict PIT guarantees.

    Attributes
    ----------
    prices : pd.DataFrame
        Adjusted close prices, index=date, columns=asset_id. Includes delisted
        names (values become NaN after delisting).
    returns : pd.DataFrame
        Simple returns derived from `prices`.
    fundamentals : dict[str, pd.DataFrame]
        Mapping of fundamental field -> (date x asset) frame, already lagged.
    metadata : pd.DataFrame
        Static/slow-moving attributes (sector, country, listing/delisting dates).
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    fundamentals: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted ascending by date")
        if self.prices.index.has_duplicates:
            raise ValueError("prices index contains duplicate dates")

    @property
    def universe(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> PanelData:
        """Return a date-bounded copy. End-inclusive."""
        mask = (self.prices.index >= start) & (self.prices.index <= end)
        return PanelData(
            prices=self.prices.loc[mask],
            returns=self.returns.loc[mask],
            fundamentals={k: v.loc[v.index.to_series().between(start, end)]
                          for k, v in self.fundamentals.items()},
            metadata=self.metadata,
        )


class PITDataLoader:
    """Loads a panel applying point-in-time corrections.

    Parameters
    ----------
    fundamental_lag_days : int
        Calendar-day lag applied to fundamental data to emulate the gap between
        period end and public availability. 90 days is a conservative default
        for quarterly reports (most are filed within ~45-90 days).
    """

    def __init__(self, fundamental_lag_days: int = 90) -> None:
        self.fundamental_lag_days = fundamental_lag_days

    def from_frames(
        self,
        prices: pd.DataFrame,
        fundamentals: dict[str, pd.DataFrame] | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> PanelData:
        """Build a PanelData from in-memory frames, applying the PIT lag.

        Fundamentals are shifted forward by `fundamental_lag_days` so that on
        any date `t`, only information that was publicly known by `t` is used.
        """
        prices = prices.sort_index()
        returns = prices.pct_change()

        lagged_fund: dict[str, pd.DataFrame] = {}
        if fundamentals:
            for name, frame in fundamentals.items():
                frame = frame.sort_index()
                shifted = frame.copy()
                # Shift the *index* forward by the reporting lag.
                shifted.index = shifted.index + timedelta(
                    days=self.fundamental_lag_days
                )
                # Reindex onto the price calendar with forward-fill (a reported
                # value remains the latest known until the next report).
                shifted = shifted.reindex(prices.index, method="ffill")
                lagged_fund[name] = shifted
                logger.info("Applied %dd PIT lag to fundamental '%s'",
                            self.fundamental_lag_days, name)

        if metadata is None:
            metadata = pd.DataFrame(index=prices.columns)

        return PanelData(
            prices=prices,
            returns=returns,
            fundamentals=lagged_fund,
            metadata=metadata,
        )

    def from_parquet(self, root: str | Path) -> PanelData:
        """Load a panel from a partitioned parquet store.

        Expected layout::

            root/prices.parquet
            root/fundamentals/<field>.parquet
            root/metadata.parquet
        """
        root = Path(root)
        prices = pd.read_parquet(root / "prices.parquet")
        prices.index = pd.to_datetime(prices.index)

        fundamentals: dict[str, pd.DataFrame] = {}
        fund_dir = root / "fundamentals"
        if fund_dir.exists():
            for fp in sorted(fund_dir.glob("*.parquet")):
                f = pd.read_parquet(fp)
                f.index = pd.to_datetime(f.index)
                fundamentals[fp.stem] = f

        metadata = pd.DataFrame()
        meta_fp = root / "metadata.parquet"
        if meta_fp.exists():
            metadata = pd.read_parquet(meta_fp)

        return self.from_frames(prices, fundamentals, metadata)


def make_synthetic_panel(
    n_assets: int = 500,
    n_days: int = 2000,
    seed: int = 42,
    delisting_rate: float = 0.05,
) -> PanelData:
    """Generate a synthetic multi-asset panel for demos and tests.

    Embeds a *real* (if weak) cross-sectional signal so that the rest of the
    pipeline has something to find: assets with higher synthetic 'quality'
    earn a small positive drift. This lets the backtest produce a non-trivial
    but realistic Sharpe rather than pure noise.

    Survivorship: a fraction of names are delisted partway through, after which
    their prices become NaN. They are NOT dropped from the panel.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    assets = [f"A{i:04d}" for i in range(n_assets)]

    # Latent quality factor -> small expected drift (the planted alpha).
    quality = rng.normal(0, 1, n_assets)
    drift = 0.0002 * quality  # ~5bp/day spread top-to-bottom

    vol = rng.uniform(0.01, 0.03, n_assets)
    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.5, 1.5, n_assets)

    rets = (
        market[:, None] * beta[None, :]
        + drift[None, :]
        + rng.normal(0, 1, (n_days, n_assets)) * vol[None, :]
    )

    # Delisting: knock out a random subset after a random date.
    n_delist = int(delisting_rate * n_assets)
    delisted = rng.choice(n_assets, size=n_delist, replace=False)
    for j in delisted:
        kill = rng.integers(n_days // 2, n_days)
        rets[kill:, j] = np.nan

    prices = pd.DataFrame(100 * np.exp(np.nancumsum(np.nan_to_num(rets), axis=0)),
                          index=dates, columns=assets)
    # Re-impose NaN after delisting on prices too.
    for j in delisted:
        first_nan = np.where(np.isnan(rets[:, j]))[0]
        if len(first_nan):
            prices.iloc[first_nan[0]:, j] = np.nan

    sectors = rng.choice(["TECH", "FIN", "ENER", "HLTH", "INDU"], n_assets)
    fundamentals = {
        # 'value' style fundamental correlated with the planted quality.
        "earnings_yield": pd.DataFrame(
            0.05 + 0.02 * quality[None, :] + rng.normal(0, 0.01, (n_days, n_assets)),
            index=dates, columns=assets,
        ),
        "book_to_price": pd.DataFrame(
            0.4 + 0.1 * quality[None, :] + rng.normal(0, 0.05, (n_days, n_assets)),
            index=dates, columns=assets,
        ),
    }
    metadata = pd.DataFrame({"sector": sectors}, index=assets)

    loader = PITDataLoader(fundamental_lag_days=0)  # synthetic already aligned
    panel = loader.from_frames(prices, fundamentals, metadata)
    logger.info("Synthetic panel: %d assets x %d days, %d delisted",
                n_assets, n_days, n_delist)
    return panel
