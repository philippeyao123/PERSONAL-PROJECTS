"""
Chargement des données de futures sur matières premières.

Source primaire : Yahoo Finance (futures front-month continus).
Fallback : générateur synthétique (Heston-like) si l'API est indisponible,
afin que le pipeline reste exécutable en toutes circonstances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_prices(tickers: list[str], start: str, end: str | None = None,
                use_synthetic_fallback: bool = True) -> pd.DataFrame:
    """Télécharge les prix de clôture ajustés pour l'univers de commodities.

    Returns
    -------
    pd.DataFrame
        Index : dates, colonnes : tickers, valeurs : prix de clôture.
    """
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, progress=False,
                          auto_adjust=True)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.dropna(how="all")
        # On exige au moins 80 % d'historique valide par actif
        valid = prices.columns[prices.notna().mean() > 0.8]
        prices = prices[valid].ffill(limit=5)
        # Neutralise les prix non positifs (ex. WTI à -$37.63 le 20/04/2020,
        # artefact du roll front-month inexploitable en continu)
        prices = prices.where(prices > 0).ffill(limit=3)
        if prices.shape[1] < 5:
            raise ValueError("Trop peu d'actifs valides téléchargés.")
        return prices
    except Exception as exc:  # noqa: BLE001
        if not use_synthetic_fallback:
            raise
        print(f"[data_loader] Téléchargement échoué ({exc}) — "
              f"bascule sur données synthétiques.")
        return _synthetic_prices(tickers, start, end)


def _synthetic_prices(tickers: list[str], start: str,
                      end: str | None) -> pd.DataFrame:
    """Génère des trajectoires GBM à vol stochastique avec corrélation
    sectorielle, pour garantir la reproductibilité hors-ligne."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start, end or pd.Timestamp.today())
    n, m = len(dates), len(tickers)

    # Corrélation bloc-sectorielle approximative
    corr = np.full((m, m), 0.25)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    # Vol stochastique (CIR discret simplifié)
    kappa, theta, xi = 3.0, 0.30**2, 0.4
    v = np.full(m, theta)
    log_p = np.log(rng.uniform(20, 100, m))
    out = np.empty((n, m))
    dt = 1 / 252
    for t in range(n):
        z = chol @ rng.standard_normal(m)
        zv = rng.standard_normal(m)
        v = np.abs(v + kappa * (theta - v) * dt + xi * np.sqrt(v * dt) * zv)
        drift = rng.normal(0.02, 0.04, m) * dt  # drifts hétérogènes
        log_p += drift - 0.5 * v * dt + np.sqrt(v * dt) * z
        out[t] = np.exp(log_p)
    return pd.DataFrame(out, index=dates, columns=tickers)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements simples journaliers (cohérents avec un PnL w * r)."""
    return prices.pct_change(fill_method=None)
