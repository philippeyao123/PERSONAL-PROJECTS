"""Critical replication: Time-Series Momentum (Moskowitz, Ooi & Pedersen, 2012).

The paper's central claim: an asset's own past 12-month excess return predicts
its next-month return, across asset classes. A strategy that goes long winners
/ short losers, scaled to constant volatility, earns a high Sharpe with low
correlation to traditional factors.

This module does three things, in increasing order of what actually
distinguishes a researcher from a paper-reader:

    1. REPLICATE   - build the 12-1 TSMOM signal, vol-scale it, measure Sharpe.
    2. STRESS      - re-measure NET of transaction costs.
    3. CRITICIZE   - split in-sample (paper era, pre-2012) vs out-of-sample
                     (post-publication), and per-decade, to test whether the
                     effect survived publication. A large literature (e.g.
                     subsequent CTA underperformance) suggests TSMOM decayed
                     materially after ~2012; the honest finding is usually
                     "the in-sample Sharpe does not persist out-of-sample."

We use liquid futures proxies (ETFs) so the study is reproducible without a
futures database. The qualitative conclusion - decay post-publication - is
robust to this approximation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Cross-asset liquid proxies: equity, bond, commodity, FX.
TSMOM_PROXIES = {
    "SPY": "equity_us",
    "EFA": "equity_intl",
    "EEM": "equity_em",
    "IEF": "bond_7_10y",
    "TLT": "bond_20y",
    "GLD": "commodity_gold",
    "DBC": "commodity_broad",
    "USO": "commodity_oil",
    "UUP": "fx_usd",
    "FXE": "fx_eur",
}


@dataclass
class TSMOMResult:
    monthly_returns: pd.Series
    gross_sharpe: float
    net_sharpe: float
    by_period: pd.DataFrame  # in/out-of-sample and per-decade breakdown


class TimeSeriesMomentum:
    """Replicate and stress-test the MOP (2012) TSMOM strategy.

    Parameters
    ----------
    lookback_months : int
        Formation window for the momentum signal (paper uses 12).
    vol_target : float
        Annualized volatility target for position scaling (paper-style).
    vol_window_months : int
        Window for the ex-ante volatility estimate used in scaling.
    cost_bps : float
        Round-trip transaction cost in bps applied on position changes.
    """

    def __init__(
        self,
        lookback_months: int = 12,
        vol_target: float = 0.40,
        vol_window_months: int = 12,
        cost_bps: float = 10.0,
    ) -> None:
        self.lookback_months = lookback_months
        self.vol_target = vol_target
        self.vol_window_months = vol_window_months
        self.cost_bps = cost_bps

    def _to_monthly(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.resample("ME").last()

    def run(self, prices: pd.DataFrame) -> TSMOMResult:
        """Run the full replicate -> stress -> criticize study.

        `prices` is a (date x asset) frame of adjusted prices for the proxies.
        """
        monthly = self._to_monthly(prices)
        rets = monthly.pct_change()

        # Ex-ante annualized vol (rolling), lagged so it is known at formation.
        vol = rets.rolling(self.vol_window_months).std() * np.sqrt(12)
        vol = vol.shift(1)

        # 12-month formation signal (sign of trailing return), lagged.
        formation = monthly.pct_change(self.lookback_months).shift(1)
        signal = np.sign(formation)

        # Vol-scaled positions: target / ex-ante vol, capped for stability.
        scale = (self.vol_target / vol).clip(upper=5.0)
        positions = signal * scale

        # Strategy return: position * next-month asset return, averaged across
        # the cross-section (equal risk budget per asset).
        gross = (positions * rets).mean(axis=1)

        # Costs: charge on absolute change in position (turnover) per asset.
        turnover = positions.diff().abs().mean(axis=1)
        cost = turnover * (self.cost_bps / 1e4)
        net = gross - cost

        gross, net = gross.dropna(), net.dropna()

        gross_sharpe = self._sharpe(gross)
        net_sharpe = self._sharpe(net)

        by_period = self._period_breakdown(net)

        logger.info("TSMOM gross Sharpe %.2f | net Sharpe %.2f",
                    gross_sharpe, net_sharpe)
        return TSMOMResult(
            monthly_returns=net,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            by_period=by_period,
        )

    @staticmethod
    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 3 or r.std(ddof=1) == 0:
            return 0.0
        return float(r.mean() / r.std(ddof=1) * np.sqrt(12))

    def _period_breakdown(self, net: pd.Series) -> pd.DataFrame:
        """In-sample (<=2012) vs out-of-sample (>2012), plus per-decade."""
        rows = []
        pub = pd.Timestamp("2012-12-31")

        insample = net[net.index <= pub]
        oos = net[net.index > pub]
        rows.append({"period": "in-sample (<=2012)",
                     "sharpe": self._sharpe(insample), "n_months": len(insample)})
        rows.append({"period": "out-of-sample (>2012)",
                     "sharpe": self._sharpe(oos), "n_months": len(oos)})

        for decade_start in [2000, 2010, 2020]:
            d0 = pd.Timestamp(f"{decade_start}-01-01")
            d1 = pd.Timestamp(f"{decade_start + 9}-12-31")
            seg = net[(net.index >= d0) & (net.index <= d1)]
            if len(seg) > 6:
                rows.append({"period": f"{decade_start}s",
                             "sharpe": self._sharpe(seg), "n_months": len(seg)})

        return pd.DataFrame(rows)


def load_tsmom_proxies(
    start: str = "2006-01-01", end: str | None = None,
    cache_root: str = "data/tsmom",
) -> pd.DataFrame:
    """Download (and cache) the ETF proxy prices for the TSMOM study."""
    from pathlib import Path

    import yfinance as yf

    path = Path(cache_root) / "proxies.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    tickers = list(TSMOM_PROXIES.keys())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)
    prices = raw["Close"].dropna(axis=1, how="all").sort_index()
    prices.index = pd.to_datetime(prices.index)

    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices
