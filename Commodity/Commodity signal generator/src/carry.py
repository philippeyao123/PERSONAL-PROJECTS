"""
Facteur carry / basis (roll yield) — extension du signal generator.

Le carry est la prime la plus documentée de la classe d'actifs
(Gorton & Rouwenhorst 2006 ; Erb & Harvey 2006 ; Koijen, Moskowitz,
Pedersen & Vrugt 2018) : long les courbes en backwardation, short les
courbes en contango.

Contrainte de données : les contrats expirés sont délistés des sources
gratuites (Yahoo), et les séries de 2e contrat continu (CHRIS) exigent
une clé API. Ce module fournit donc :

  1. LIVE  — construction de la courbe à terme complète depuis les
     contrats individuels listés sur Yahoo, et calcul du carry courant.
  2. PLUG-READY — `carry_signal_from_history(f1, f2, dt_years)` pour
     backtester le sleeve dès que des séries F1/F2 historiques sont
     disponibles (Nasdaq Data Link CHRIS, Bloomberg, Refinitiv...).

Méthodologie du carry live :
  - Prioritaire : paire de contrats à ~12 mois d'écart sur le MÊME mois
    calendaire, ce qui neutralise la saisonnalité de la courbe — point
    critique pour NG et les agricoles, où la pente calendaire brute
    confond saisonnalité et prime de risque.
  - Fallback : pente de la régression ln(F) ~ maturité sur tous les
    contrats listés < 14 mois (au moins 3 points).
  - Convention de signe : carry > 0 = backwardation (signal long).
"""

from __future__ import annotations

import calendar
import datetime as dt

import numpy as np
import pandas as pd

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

# Suffixe Yahoo par place de cotation
EXCHANGE_SUFFIX = {
    "CL": "NYM", "BZ": "NYM", "NG": "NYM", "HO": "NYM", "RB": "NYM",
    "PL": "NYM",
    "GC": "CMX", "SI": "CMX", "HG": "CMX",
    "ZC": "CBT", "ZW": "CBT", "ZS": "CBT",
    "SB": "NYB", "KC": "NYB", "CT": "NYB",
}

# Cycles de maturités listées (codes mois) par racine
LISTED_CYCLES = {
    "CL": "FGHJKMNQUVXZ", "BZ": "FGHJKMNQUVXZ", "NG": "FGHJKMNQUVXZ",
    "HO": "FGHJKMNQUVXZ", "RB": "FGHJKMNQUVXZ",
    "GC": "GJMQVZ", "SI": "FHKNUZ", "HG": "FHKNUZ", "PL": "FJNV",
    "ZC": "HKNUZ", "ZW": "HKNUZ", "ZS": "FHKNQUX",
    "SB": "HKNV", "KC": "HKNUZ", "CT": "HKNVZ",
}


def _contract_chain(root: str, today: dt.date,
                    max_months: int = 14) -> list[tuple[str, dt.date]]:
    """Liste (ticker Yahoo, date de maturité approx.) des contrats listés
    de `root` expirant dans (1, max_months] mois."""
    cycle = LISTED_CYCLES[root]
    suffix = EXCHANGE_SUFFIX[root]
    out = []
    for k in range(1, max_months + 1):
        y, m = divmod(today.month - 1 + k, 12)
        year, month = today.year + y, m + 1
        code = MONTH_CODES[month]
        if code not in cycle:
            continue
        # Maturité approximée au milieu du mois de contrat — suffisant
        # pour annualiser un écart de ~12 mois (erreur < 5 %).
        maturity = dt.date(year, month, 15)
        out.append((f"{root}{code}{str(year)[-2:]}.{suffix}", maturity))
    return out


def fetch_curves(roots: list[str],
                 today: dt.date | None = None) -> dict[str, pd.Series]:
    """Télécharge les courbes à terme live : {racine: Series(prix,
    index=maturités)}. Les contrats sans cotation sont ignorés."""
    import yfinance as yf

    today = today or dt.date.today()
    chains = {r: _contract_chain(r, today) for r in roots}
    all_tickers = [t for ch in chains.values() for t, _ in ch]

    data = yf.download(all_tickers, period="5d", progress=False,
                       auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    last = close.ffill().iloc[-1]

    curves = {}
    for root, chain in chains.items():
        pts = {mat: float(last[tk]) for tk, mat in chain
               if tk in last.index and np.isfinite(last[tk])
               and last[tk] > 0}
        if len(pts) >= 3:
            curves[root] = pd.Series(pts).sort_index()
    return curves


def carry_from_curve(curve: pd.Series, today: dt.date | None = None,
                     seasonal_pair_months: int = 12) -> tuple[float, str]:
    """Carry annualisé d'une courbe.

    Prioritaire : paire (contrat le plus proche, contrat ~12 mois plus
    loin sur le même mois calendaire) → neutralise la saisonnalité.
    Fallback : pente de ln(F) ~ maturité (années) sur toute la courbe.

    Returns (carry annualisé, méthode utilisée).
    """
    today = today or dt.date.today()
    mats = list(curve.index)
    near = mats[0]

    # Paire saisonnière : même mois calendaire, ~12 mois d'écart
    for far in mats[1:]:
        gap = (far.year - near.year) * 12 + (far.month - near.month)
        if far.month == near.month and abs(gap - seasonal_pair_months) <= 1:
            dt_years = gap / 12
            carry = np.log(curve[near] / curve[far]) / dt_years
            return float(carry), "seasonal_pair_12m"

    # Fallback : pente de régression
    t = np.array([(m - today).days / 365.25 for m in mats])
    slope = np.polyfit(t, np.log(curve.to_numpy()), 1)[0]
    return float(-slope), "curve_slope"


def live_carry_signal(universe: dict[str, str]) -> pd.DataFrame:
    """Signal carry cross-sectionnel courant pour l'univers.

    Returns DataFrame indexé par ticker continu Yahoo (ex. 'CL=F') :
    carry annualisé, méthode, n points de courbe, score XS dans [-1, 1].
    """
    roots = [t.replace("=F", "") for t in universe]
    curves = fetch_curves(roots)

    rows = {}
    for root, curve in curves.items():
        carry, method = carry_from_curve(curve)
        rows[f"{root}=F"] = {"carry_annualized": carry, "method": method,
                             "n_contracts": len(curve)}
    df = pd.DataFrame(rows).T
    if df.empty:
        return df

    rank = df["carry_annualized"].rank()
    n = rank.count()
    df["carry_signal"] = ((rank - (n + 1) / 2) / ((n - 1) / 2)).round(3)
    df["state"] = np.where(df["carry_annualized"] > 0,
                           "backwardation", "contango")
    return df.sort_values("carry_annualized", ascending=False), curves


# ---------------------------------------------------------------------------
# Backtest historique — plug-ready
# ---------------------------------------------------------------------------
def carry_signal_from_history(f1: pd.DataFrame, f2: pd.DataFrame,
                              dt_years: float | pd.DataFrame) -> pd.DataFrame:
    """Signal carry historique à partir de séries F1/F2 (dates x actifs).

    carry_t = ln(F1_t / F2_t) / dt_years, puis rang cross-sectionnel
    dans [-1, 1]. À brancher dans build_composite (via config) dès que
    des données de courbe historiques sont disponibles.

    Parameters
    ----------
    f1, f2 : prix du 1er et 2e contrat continus (mêmes index/colonnes).
    dt_years : écart de maturité en années (scalaire ou DataFrame).
    """
    carry = np.log(f1 / f2) / dt_years
    rank = carry.rank(axis=1)
    n = rank.count(axis=1)
    score = rank.sub((n + 1) / 2, axis=0).div((n - 1) / 2, axis=0)
    return score.where(carry.notna())
