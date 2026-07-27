"""Portfolio performance and risk measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from systematic_research.exceptions import DataValidationError


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty or periods_per_year <= 0:
        return float("nan")
    wealth = float((1.0 + clean).prod())
    if wealth <= 0:
        return -1.0
    return wealth ** (periods_per_year / len(clean)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    if not 0 < confidence < 1:
        raise DataValidationError("VaR confidence must be between zero and one")
    return float(-returns.dropna().quantile(1.0 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    var = value_at_risk(returns, confidence)
    tail = returns.dropna()[returns.dropna() <= -var]
    return float(-tail.mean()) if not tail.empty else var


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 252, risk_free_return: float = 0.0
) -> float:
    excess = returns.dropna() - risk_free_return
    volatility = excess.std(ddof=1)
    return float(excess.mean() / volatility * np.sqrt(periods_per_year)) if volatility > 0 else 0.0


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0].std(ddof=1)
    return float(clean.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    drawdown = abs(max_drawdown(returns))
    return annualized_return(returns, periods_per_year) / drawdown if drawdown > 0 else 0.0


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    return {
        "total_return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns),
        "expected_shortfall_95": expected_shortfall(returns),
    }


@dataclass(frozen=True)
class ExposureSnapshot:
    gross: float
    net: float
    long: float
    short: float
    concentration: float


def exposure_snapshot(weights: pd.Series) -> ExposureSnapshot:
    return ExposureSnapshot(
        gross=float(weights.abs().sum()),
        net=float(weights.sum()),
        long=float(weights.clip(lower=0).sum()),
        short=float(weights.clip(upper=0).sum()),
        concentration=float(weights.abs().max()) if len(weights) else 0.0,
    )
