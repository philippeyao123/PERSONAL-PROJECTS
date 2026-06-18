"""
Configuration centrale du Commodity Trading Signal Generator.
"""

# ---------------------------------------------------------------------------
# Univers de commodities (futures front-month, tickers Yahoo Finance)
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Energy
    "CL=F": "WTI Crude Oil",
    "BZ=F": "Brent Crude Oil",
    "NG=F": "Natural Gas",
    "HO=F": "Heating Oil",
    "RB=F": "RBOB Gasoline",
    # Metals
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "PL=F": "Platinum",
    # Agriculture
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
    "ZS=F": "Soybeans",
    "SB=F": "Sugar",
    "KC=F": "Coffee",
    "CT=F": "Cotton",
}

SECTORS = {
    "CL=F": "Energy", "BZ=F": "Energy", "NG=F": "Energy",
    "HO=F": "Energy", "RB=F": "Energy",
    "GC=F": "Metals", "SI=F": "Metals", "HG=F": "Metals", "PL=F": "Metals",
    "ZC=F": "Agriculture", "ZW=F": "Agriculture", "ZS=F": "Agriculture",
    "SB=F": "Softs", "KC=F": "Softs", "CT=F": "Softs",
}

# ---------------------------------------------------------------------------
# Période de backtest
# ---------------------------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = None  # None = aujourd'hui

# ---------------------------------------------------------------------------
# Paramètres des signaux
# ---------------------------------------------------------------------------
SIGNALS_CONFIG = {
    "tsmom": {
        "lookbacks": [252],            # 12 mois, sign-based (Moskowitz et al. 2012)
        "weight": 0.40,
    },
    "xsmom": {
        "lookback": 252,               # momentum 12m-1m (Miffre & Rallis 2007)
        "skip": 21,
        "weight": 0.10,
    },
    "short_term_reversal": {
        "lookback": 5,                 # réversion hebdomadaire vol-ajustée
        "vol_window": 21,
        "weight": 0.15,
    },
    "vol_breakout": {
        "channel": 55,                 # canal Donchian intermédiaire
        "weight": 0.20,
    },
    "skewness": {
        "lookback": 252,               # Fernandez-Perez et al. (2018)
        "weight": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Portefeuille & exécution
# ---------------------------------------------------------------------------
PORTFOLIO_CONFIG = {
    "vol_target_portfolio": 0.10,  # vol annualisée cible du portefeuille
    "vol_lookback": 63,            # fenêtre EWMA pour la vol par actif
    "cov_lambda": 0.97,            # lambda RiskMetrics (covariance EWMA)
    "warmup": 252,                 # pas de trading avant 1 an d'historique
    "max_position": 0.60,          # cap de position par actif (x notionnel)
    "max_gross_leverage": 5.0,     # levier brut max (cap de sécurité)
    "no_trade_band": 0.01,         # bande de non-trading (1% notionnel)
    "transaction_cost_bps": 3.0,   # coûts de transaction (aller simple)
    "execution_lag": 1,            # exécution t+1 (pas de look-ahead)
}

ANNUALIZATION = 252
RISK_FREE = 0.0
