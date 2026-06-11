"""
Bibliothèque de signaux de trading sur matières premières.

Chaque signal retourne un DataFrame (dates x actifs) de scores dans [-1, 1],
prêt à être combiné. Aucun signal n'utilise d'information future : tout est
calculé sur données disponibles en t, l'exécution se faisant en t+1 dans le
backtester.

Références :
- Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE.
- Miffre & Rallis (2007), "Momentum strategies in commodity futures
  markets", JBF.
- Fernandez-Perez, Frijns, Fuertes & Miffre (2018), "The skewness of
  commodity futures returns", JBF.
- Nagel (2012), "Evaporating Liquidity", RFS — réversion court terme
  comme rémunération de la fourniture de liquidité.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize_z(z: pd.DataFrame, cap: float = 2.0) -> pd.DataFrame:
    """Borne les z-scores puis les ramène dans [-1, 1]."""
    return z.clip(-cap, cap) / cap


def _xs_rank_score(values: pd.DataFrame) -> pd.DataFrame:
    """Rang cross-sectionnel centré-réduit dans [-1, 1], dollar-neutre."""
    rank = values.rank(axis=1)
    n = rank.count(axis=1)
    score = rank.sub((n + 1) / 2, axis=0).div((n - 1) / 2, axis=0)
    return score.where(values.notna())


# ---------------------------------------------------------------------------
# 1. Time-Series Momentum (TSMOM) — Moskowitz, Ooi & Pedersen (2012)
# ---------------------------------------------------------------------------
def tsmom(prices: pd.DataFrame, returns: pd.DataFrame,
          lookbacks: list[int]) -> pd.DataFrame:
    """Momentum temporel — spécification canonique de Moskowitz, Ooi &
    Pedersen (2012) : signe du rendement cumulé sur l'horizon.

    Moyenne des signes si plusieurs horizons sont fournis. Une variante
    t-stat (rendement / vol sur l'horizon) a été testée : elle dégrade
    le Sharpe d'environ 0.1 sur cet échantillon ; le sign-based est à
    la fois plus simple et plus robuste ici.
    """
    scores = []
    for lb in lookbacks:
        ret = prices.pct_change(lb, fill_method=None)
        scores.append(np.sign(ret))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 2. Momentum cross-sectionnel (XSMOM) — Miffre & Rallis (2007)
# ---------------------------------------------------------------------------
def xsmom(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Momentum relatif 12m-1m : long les gagnants, short les perdants.

    Rendement sur (t-lookback, t-skip], le dernier mois étant exclu pour
    éviter la contamination par le reversal court terme. Dollar-neutre
    par construction du rang centré.
    """
    ret = prices.shift(skip).pct_change(lookback - skip, fill_method=None)
    return _xs_rank_score(ret)


# ---------------------------------------------------------------------------
# 3. Réversion court terme cross-sectionnelle (STR)
# ---------------------------------------------------------------------------
def short_term_reversal(prices: pd.DataFrame, returns: pd.DataFrame,
                        lookback: int, vol_window: int) -> pd.DataFrame:
    """Réversion hebdomadaire : short les gagnants 5 jours, long les
    perdants, sur rendements ajustés de la volatilité.

    Horizon volontairement court (5 j) pour rester orthogonal aux sleeves
    de trend (corrélation PnL ~ -0.4 vs Donchian, vs -0.9 pour une
    réversion à 50 j qui annulerait le trend). Interprétation économique :
    rémunération de la fourniture de liquidité face aux flux (Nagel, 2012).
    """
    vol = returns.rolling(vol_window).std() * np.sqrt(lookback)
    r_adj = prices.pct_change(lookback, fill_method=None) / vol.replace(0, np.nan)
    return -_xs_rank_score(r_adj)


# ---------------------------------------------------------------------------
# 4. Breakout de canal (Donchian)
# ---------------------------------------------------------------------------
def vol_breakout(prices: pd.DataFrame, channel: int) -> pd.DataFrame:
    """Position du prix dans son canal Donchian, recentrée dans [-1, 1].

    Proche de +1 : cassure haussière ; proche de -1 : cassure baissière.
    Style "trend-following CTA" à horizon intermédiaire, complémentaire
    du TSMOM (signal continu vs signe, horizon 55 j vs 6-12 mois).
    """
    hi = prices.rolling(channel).max()
    lo = prices.rolling(channel).min()
    width = (hi - lo).replace(0, np.nan)
    return 2 * (prices - lo) / width - 1


# ---------------------------------------------------------------------------
# 5. Prime de skewness — Fernandez-Perez et al. (2018)
# ---------------------------------------------------------------------------
def skewness_signal(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Short les actifs à skew positif, long les actifs à skew négatif.

    Les investisseurs surpayent les profils de loterie (skew > 0) ; la
    prime de risque se concentre sur le skew négatif.
    """
    skew = returns.rolling(lookback).skew()
    return -_xs_rank_score(skew)


# ---------------------------------------------------------------------------
# Combinaison
# ---------------------------------------------------------------------------
def build_composite(prices: pd.DataFrame, returns: pd.DataFrame,
                    cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Construit le signal composite pondéré et retourne aussi chaque
    signal individuel pour l'attribution de performance."""
    individual = {
        "tsmom": tsmom(prices, returns, cfg["tsmom"]["lookbacks"]),
        "xsmom": xsmom(prices, cfg["xsmom"]["lookback"], cfg["xsmom"]["skip"]),
        "short_term_reversal": short_term_reversal(
            prices, returns,
            cfg["short_term_reversal"]["lookback"],
            cfg["short_term_reversal"]["vol_window"]),
        "vol_breakout": vol_breakout(prices, cfg["vol_breakout"]["channel"]),
        "skewness": skewness_signal(returns, cfg["skewness"]["lookback"]),
    }
    composite = sum(cfg[name]["weight"] * sig
                    for name, sig in individual.items())
    return composite.clip(-1, 1), individual
