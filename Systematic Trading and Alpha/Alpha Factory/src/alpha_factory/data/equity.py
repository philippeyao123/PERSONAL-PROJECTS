"""Real equity panel loader.

Fetches adjusted prices and volume for a configurable universe via yfinance,
derives a dollar-ADV series (needed for honest capacity / impact work), and
caches everything to a partitioned parquet store so subsequent runs are fast
and offline-reproducible.

Survivorship note: yfinance only serves *currently listed* tickers, so a
universe built from today's index membership is survivorship-biased by
construction. This loader does not pretend otherwise — it exposes a
`survivorship_warning` flag and the README/replication study discuss the
implication. For a fully bias-free study you need a point-in-time membership
and delisting database (CRSP); the architecture here accepts such a source
unchanged because everything downstream consumes `PanelData`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from alpha_factory.data.loader import PanelData, PITDataLoader

logger = logging.getLogger(__name__)


# A compact, liquid, sector-diversified default universe (large-cap US).
# Deliberately small so demos run fast; swap for an index membership list.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO",  # tech
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",                      # financials
    "XOM", "CVX", "COP", "SLB", "EOG",                                # energy
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",                        # health
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE",                   # staples/disc
    "CAT", "BA", "HON", "GE", "UPS", "LMT",                           # industrials
    "LIN", "SHW",                                                     # materials
    "NEE", "DUK", "SO",                                              # utilities
    "AMT", "PLD",                                                    # real estate
    "DIS", "NFLX", "CMCSA", "T", "VZ",                               # comm services
]


class EquityDataLoader:
    """Loads a real equity panel, with a transparent parquet cache.

    Parameters
    ----------
    cache_root : str | Path
        Directory for the parquet cache (created if missing).
    fundamental_lag_days : int
        Reporting lag for any fundamentals (passed to PITDataLoader).
    """

    def __init__(
        self, cache_root: str | Path = "data/equity",
        fundamental_lag_days: int = 90,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.pit = PITDataLoader(fundamental_lag_days=fundamental_lag_days)
        self.survivorship_warning = True

    def _cache_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.cache_root / "prices.parquet",
            self.cache_root / "dollar_adv.parquet",
            self.cache_root / "metadata.parquet",
        )

    def load(
        self,
        tickers: list[str] | None = None,
        start: str = "2015-01-01",
        end: str | None = None,
        use_cache: bool = True,
        adv_window: int = 63,
    ) -> tuple[PanelData, pd.DataFrame]:
        """Return (PanelData, dollar_adv).

        dollar_adv is a (date x asset) frame of trailing-average dollar volume,
        used by capacity analysis for realistic participation rates.
        """
        tickers = tickers or DEFAULT_UNIVERSE
        px_path, adv_path, meta_path = self._cache_paths()

        if use_cache and px_path.exists() and adv_path.exists():
            logger.info("Loading equity panel from cache %s", self.cache_root)
            prices = pd.read_parquet(px_path)
            prices.index = pd.to_datetime(prices.index)
            dollar_adv = pd.read_parquet(adv_path)
            dollar_adv.index = pd.to_datetime(dollar_adv.index)
            metadata = (pd.read_parquet(meta_path) if meta_path.exists()
                        else pd.DataFrame(index=prices.columns))
        else:
            prices, dollar_adv, metadata = self._download(
                tickers, start, end, adv_window
            )
            self.cache_root.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(px_path)
            dollar_adv.to_parquet(adv_path)
            metadata.to_parquet(meta_path)
            logger.info("Cached equity panel to %s", self.cache_root)

        if self.survivorship_warning:
            logger.warning(
                "Universe is built from currently-listed tickers: "
                "SURVIVORSHIP BIAS present. Use a PIT membership source for "
                "bias-free research."
            )

        panel = self.pit.from_frames(prices, fundamentals=None, metadata=metadata)
        return panel, dollar_adv

    def _download(
        self, tickers: list[str], start: str, end: str | None, adv_window: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        import yfinance as yf

        logger.info("Downloading %d tickers from yfinance...", len(tickers))
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False,
        )
        prices = raw["Close"].copy()
        volume = raw["Volume"].copy()

        # Drop tickers that returned no data at all.
        prices = prices.dropna(axis=1, how="all")
        volume = volume.reindex(columns=prices.columns)

        # Dollar volume -> trailing-average ADV.
        dollar_vol = prices * volume
        dollar_adv = dollar_vol.rolling(adv_window, min_periods=adv_window // 2).mean()

        # Sector metadata (best-effort; yfinance .info is slow/rate-limited so
        # we attach a coarse static map and leave enrichment to the user).
        metadata = pd.DataFrame(index=prices.columns)
        metadata["sector"] = [_coarse_sector(t) for t in prices.columns]

        prices.index = pd.to_datetime(prices.index)
        dollar_adv.index = pd.to_datetime(dollar_adv.index)
        return prices.sort_index(), dollar_adv.sort_index(), metadata


# Minimal static sector map for the default universe (avoids slow .info calls).
_SECTOR_MAP = {
    "AAPL": "TECH", "MSFT": "TECH", "AMZN": "DISC", "GOOGL": "COMM",
    "META": "COMM", "NVDA": "TECH", "TSLA": "DISC", "AVGO": "TECH",
    "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN", "MS": "FIN",
    "C": "FIN", "AXP": "FIN", "XOM": "ENER", "CVX": "ENER", "COP": "ENER",
    "SLB": "ENER", "EOG": "ENER", "JNJ": "HLTH", "UNH": "HLTH", "PFE": "HLTH",
    "MRK": "HLTH", "ABBV": "HLTH", "LLY": "HLTH", "PG": "STPL", "KO": "STPL",
    "PEP": "STPL", "WMT": "STPL", "COST": "STPL", "MCD": "DISC", "NKE": "DISC",
    "CAT": "INDU", "BA": "INDU", "HON": "INDU", "GE": "INDU", "UPS": "INDU",
    "LMT": "INDU", "LIN": "MATL", "SHW": "MATL", "NEE": "UTIL", "DUK": "UTIL",
    "SO": "UTIL", "AMT": "RE", "PLD": "RE", "DIS": "COMM", "NFLX": "COMM",
    "CMCSA": "COMM", "T": "COMM", "VZ": "COMM",
}


def _coarse_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker, "OTHER")
