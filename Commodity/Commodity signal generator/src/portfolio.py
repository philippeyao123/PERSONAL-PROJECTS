"""
Construction de portefeuille : stack CTA standard.

  1. Book brut inverse-volatilité : w_i ∝ signal_i / sigma_i
     (chaque actif contribue un risque comparable par unité de signal).
  2. Scaling unique vers la vol cible du portefeuille, via la vol
     ex-ante sqrt(w' Σ w) avec Σ covariance EWMA (RiskMetrics).
  3. Caps de sécurité (position, levier brut) dimensionnés pour ne
     mordre qu'exceptionnellement — ils ne font pas le sizing.
  4. Bande de non-trading (Gârleanu & Pedersen, 2013, en version
     simplifiée) : on ne trade vers la cible que lorsque l'écart
     dépasse la bande, ce qui divise le turnover par ~2 en préservant
     l'alpha des signaux rapides (contrairement à un rebalancement
     hebdomadaire, qui détruit la réversion 5 jours).

Aucun look-ahead : Σ_t n'utilise que les rendements jusqu'à t inclus,
et l'exécution se fait en t+1 dans le backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Volatilité EWMA annualisée par actif."""
    lam = 1 - 2 / (lookback + 1)
    var = returns.pow(2).ewm(alpha=1 - lam, min_periods=lookback // 2).mean()
    return np.sqrt(var * 252)


def exante_portfolio_vol(weights: pd.DataFrame, returns: pd.DataFrame,
                         lam: float = 0.97,
                         warmup: int = 252) -> pd.Series:
    """Vol ex-ante annualisée du book : sigma_p(t) = sqrt(w_t' Σ_t w_t),
    Σ_t covariance EWMA (RiskMetrics) estimée sur r_{<=t}."""
    R = returns.fillna(0.0).to_numpy()
    W = weights.fillna(0.0).to_numpy()
    n, m = R.shape
    sigma = np.zeros((m, m))
    out = np.full(n, np.nan)
    for t in range(n):
        r = R[t][:, None]
        sigma = lam * sigma + (1 - lam) * (r @ r.T)
        if t >= warmup:
            out[t] = np.sqrt(max(W[t] @ sigma @ W[t], 0.0) * 252)
    return pd.Series(out, index=returns.index)


def build_positions(signals: pd.DataFrame, returns: pd.DataFrame,
                    cfg: dict) -> pd.DataFrame:
    """Transforme les scores de signal en poids de portefeuille."""
    # 1. Book brut inverse-vol (sans dimension)
    vol = ewma_vol(returns, cfg["vol_lookback"])
    raw = (signals / vol.replace(0, np.nan)).fillna(0.0)

    # 2. Scaling unique vers la vol cible (pas de trading avant warmup)
    exante = exante_portfolio_vol(raw, returns,
                                  lam=cfg["cov_lambda"],
                                  warmup=cfg["warmup"])
    scaler = (cfg["vol_target_portfolio"] / exante).replace(
        [np.inf, -np.inf], np.nan)
    weights = raw.mul(scaler, axis=0).fillna(0.0)

    # 3. Caps de sécurité
    weights = weights.clip(-cfg["max_position"], cfg["max_position"])
    gross = weights.abs().sum(axis=1)
    weights = weights.div(
        (gross / cfg["max_gross_leverage"]).clip(lower=1.0), axis=0)

    # 4. Bande de non-trading : positions tenues vs cibles
    band = cfg["no_trade_band"]
    target = weights.to_numpy()
    held = np.zeros_like(target)
    h = np.zeros(target.shape[1])
    for t in range(target.shape[0]):
        dev = target[t] - h
        h = np.where(np.abs(dev) > band,
                     target[t] - np.sign(dev) * band, h)
        held[t] = h
    return pd.DataFrame(held, index=weights.index,
                        columns=weights.columns)
