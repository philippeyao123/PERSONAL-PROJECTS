"""
Data layer — continuous front-month contracts and WTI term-structure panel.

Sources (Yahoo Finance):
    CL=F  WTI front month (NYMEX)
    BZ=F  Brent front month (ICE)
    NG=F  Henry Hub natural gas front month (NYMEX)
    Individual WTI contracts (e.g. CLZ26.NYM) for the futures curve panel.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def load_front_contracts(start: str = "2015-01-01") -> pd.DataFrame:
    """Continuous front-month closes for WTI, Brent, NatGas."""
    raw = yf.download(["CL=F", "BZ=F", "NG=F"], start=start, progress=False)["Close"]
    px = raw.rename(columns={"CL=F": "WTI", "BZ=F": "Brent", "NG=F": "NatGas"})
    px = px.dropna(how="all").ffill().dropna()
    # 20 Apr 2020: WTI May-20 contract settled at -$37.63. Log-price models are
    # undefined there; standard treatment is to bridge that single print
    # (equivalently, roll to the June contract as most index providers did).
    px[px <= 0] = np.nan
    px = px.ffill()
    return px


def wti_contract_tickers(asof: dt.date, n_contracts: int = 8) -> list[tuple[str, dt.date]]:
    """
    Build (ticker, expiry) pairs for the next `n_contracts` WTI monthly contracts.
    WTI (CL) expires ~3 business days before the 25th of the month preceding
    delivery; we approximate expiry as the 20th of that month, which is
    accurate to a few days and sufficient for time-to-maturity computation.
    """
    out = []
    y, m = asof.year, asof.month
    # start from delivery month two months ahead to ensure liquidity/history
    m += 2
    while len(out) < n_contracts:
        if m > 12:
            m -= 12
            y += 1
        code = [k for k, v in MONTH_CODES.items() if v == m][0]
        ticker = f"CL{code}{str(y)[2:]}.NYM"
        em, ey = (m - 1, y) if m > 1 else (12, y - 1)
        expiry = dt.date(ey, em, 20)
        out.append((ticker, expiry))
        m += 1
    return out


def load_wti_term_structure(start: str = "2024-06-01",
                            n_contracts: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Panel of log futures prices and matching time-to-maturity (in years).

    The panel combines the continuous front month (CL=F, tau ~ 2-6 weeks)
    with `n_contracts` fixed contracts further out. Without the front month,
    early-sample maturities are all long-dated and the short-term factor is
    weakly identified (its loading e^{-kappa tau} ~ 0); the front contract
    pins it down across the whole sample.

    Returns
    -------
    log_f : DataFrame (dates x contracts) of log prices
    tau   : DataFrame (dates x contracts) of time to maturity in years
    """
    pairs = wti_contract_tickers(dt.date.today(), n_contracts)
    tickers = [p[0] for p in pairs]
    expiries = {p[0]: p[1] for p in pairs}

    px = yf.download(["CL=F"] + tickers, start=start, progress=False)["Close"]
    px = px.dropna(how="all").ffill().dropna()
    px[px <= 0] = np.nan
    px = px.ffill()

    def front_expiry(d: dt.date) -> dt.date:
        # WTI front expires ~3 business days before the 25th of the month
        # preceding delivery; approximate as the 20th. After the 20th the
        # front rolls to the following month.
        if d.day <= 19:
            return dt.date(d.year, d.month, 20)
        m, y = (d.month + 1, d.year) if d.month < 12 else (1, d.year + 1)
        return dt.date(y, m, 20)

    tau = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for col in px.columns:
        if col == "CL=F":
            tau[col] = [(front_expiry(d.date()) - d.date()).days / 365.25
                        for d in px.index]
        else:
            tau[col] = [(expiries[col] - d.date()).days / 365.25 for d in px.index]

    log_f = np.log(px)
    mask = (tau > 1 / 365).all(axis=1)
    return log_f.loc[mask], tau.loc[mask]
