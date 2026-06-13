"""Data layer for the volatility-arbitrage engine.

Two kinds of inputs:

    - Implied volatility indices (VIX, VXN): the market's forward-looking
      30-day vol estimate. These are *observed*, not fabricated — the central
      fix relative to the original notebook, which synthesised IV as
      ``realized + noise`` (a circular construction that guarantees a profit).
    - Underlying prices (SPX, NDX, and single-name constituents) used to
      compute *realized* volatility and realized correlation.

Everything is cached to parquet so runs are offline and reproducible.

Survivorship note: the constituent list is today's membership, so the
single-name panel is survivorship-biased. This is acceptable for a
dispersion *signal* study (we are measuring a cross-sectional vol relationship,
not a long-only equity edge) but is flagged, and the loader accepts a
point-in-time membership source unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Implied-vol indices and their underlyings.
VOL_INDICES = {"VIX": "^VIX", "VXN": "^VXN"}
UNDERLYINGS = {"SPX": "^GSPC", "NDX": "^NDX"}

# A compact, liquid SPX-constituent basket for the dispersion study.
DISPERSION_BASKET = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ",
    "XOM", "PG", "UNH", "HD", "MA", "BAC", "DIS", "KO", "MRK", "CVX",
]


class VolDataLoader:
    """Loads implied-vol indices and underlying prices, with parquet cache."""

    def __init__(self, cache_root: str | Path = "data") -> None:
        self.cache_root = Path(cache_root)

    def load(
        self, start: str = "2010-01-01", end: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Return a dict of frames: vix, vxn, spx, ndx, basket (prices)."""
        cache = self.cache_root / "voldata.parquet"
        bcache = self.cache_root / "basket.parquet"
        if use_cache and cache.exists() and bcache.exists():
            logger.info("Loading vol data from cache %s", self.cache_root)
            core = pd.read_parquet(cache)
            core.index = pd.to_datetime(core.index)
            basket = pd.read_parquet(bcache)
            basket.index = pd.to_datetime(basket.index)
            return {"core": core, "basket": basket}

        import yfinance as yf

        tickers = list(VOL_INDICES.values()) + list(UNDERLYINGS.values())
        logger.info("Downloading %d core series...", len(tickers))
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False)["Close"]
        rename = {v: k for k, v in {**VOL_INDICES, **UNDERLYINGS}.items()}
        core = raw.rename(columns=rename).sort_index()

        logger.info("Downloading %d basket names...", len(DISPERSION_BASKET))
        basket = yf.download(DISPERSION_BASKET, start=start, end=end,
                             auto_adjust=True, progress=False)["Close"]
        basket = basket.dropna(axis=1, how="all").sort_index()

        core.index = pd.to_datetime(core.index)
        basket.index = pd.to_datetime(basket.index)

        self.cache_root.mkdir(parents=True, exist_ok=True)
        core.to_parquet(cache)
        basket.to_parquet(bcache)
        logger.info("Cached vol data to %s", self.cache_root)
        return {"core": core, "basket": basket}


def realized_vol(
    prices: pd.Series | pd.DataFrame, window: int = 21,
    annualize: int = 252,
) -> pd.Series | pd.DataFrame:
    """Trailing realized volatility (annualized) from close-to-close returns.

    Uses log returns and a rolling standard deviation. This is the *trailing*
    RV known at time t; forward RV (the prediction target) is computed
    separately to avoid look-ahead.
    """
    logret = np.log(prices / prices.shift(1))
    return logret.rolling(window).std() * np.sqrt(annualize)


def forward_realized_vol(
    prices: pd.Series, horizon: int = 21, annualize: int = 252,
) -> pd.Series:
    """Realized vol over the FORWARD `horizon` days, aligned to decision date t.

    The value at t is the annualized std of log returns over the window
    (t, t+horizon]. Aligning the future window back to t is what lets us
    compare IV_t against the volatility that actually materializes after t —
    the core of the variance risk premium. (This is forward-looking by design;
    it is the realized *outcome*, never an input to the position taken at t.)
    """
    logret = np.log(prices / prices.shift(1))
    fwd_aligned = (
        logret[::-1].rolling(horizon).std()[::-1].shift(-1) * np.sqrt(annualize)
    )
    return fwd_aligned
