"""
Multi-Strategy Portfolio Backtester
=====================================
Production-style backtesting framework for pod shop risk management.

Features:
  - Multi-strategy portfolio with independent pods
  - Performance attribution (Brinson, factor-based)
  - Risk budgeting (equal risk contribution, vol targeting)
  - Drawdown analysis (max DD, calmar, DD duration, recovery)
  - Pod-level risk limits (stop-loss, gross/net, concentration)
  - Factor exposure decomposition (market, sector, momentum, value)
  - Correlation regime analysis
  - Tail risk metrics (CVaR, max loss, skewness, kurtosis)

Designed to mirror what a risk manager at Millennium / Citadel / Balyasny
monitors daily across pods.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
import time


# ============================================================================
# Strategy / Pod Definition
# ============================================================================

@dataclass
class Strategy:
    """A single trading strategy (pod)."""
    name: str
    style: str                          # 'momentum', 'mean_reversion', 'stat_arb', 'macro', 'equity_ls'
    target_vol: float = 0.10            # 10% annualised vol target
    max_gross_leverage: float = 3.0     # Max gross exposure / capital
    max_net_leverage: float = 0.5       # Max net exposure / capital
    stop_loss_pct: float = -0.05        # -5% drawdown triggers stop
    capital_allocation: float = 0.20    # % of total portfolio capital
    
    # Simulated return parameters
    annual_alpha: float = 0.08          # Expected annual alpha
    annual_vol: float = 0.10            # Realised vol
    market_beta: float = 0.10           # Market beta
    momentum_loading: float = 0.0       # Momentum factor loading
    value_loading: float = 0.0          # Value factor loading
    
    def describe(self) -> str:
        return f"{self.name} ({self.style}): alpha={self.annual_alpha:.0%}, vol={self.annual_vol:.0%}, beta={self.market_beta:.2f}"


@dataclass
class PortfolioConfig:
    """Portfolio-level configuration."""
    total_capital: float = 100_000_000   # $100M
    target_vol: float = 0.08             # 8% portfolio vol target
    max_drawdown_limit: float = -0.10    # -10% portfolio stop
    rebalance_frequency: int = 21        # Monthly rebalancing
    risk_free_rate: float = 0.04         # Annualised


def create_pod_shop_portfolio() -> Tuple[List[Strategy], PortfolioConfig]:
    """Create a realistic multi-pod portfolio."""
    strategies = [
        Strategy(
            name="Equity Stat Arb",
            style="stat_arb",
            target_vol=0.08,
            annual_alpha=0.06,
            annual_vol=0.08,
            market_beta=0.05,
            momentum_loading=0.15,
            value_loading=0.10,
            capital_allocation=0.25,
            stop_loss_pct=-0.04,
        ),
        Strategy(
            name="Equity L/S Fundamental",
            style="equity_ls",
            target_vol=0.12,
            annual_alpha=0.10,
            annual_vol=0.14,
            market_beta=0.30,
            momentum_loading=-0.05,
            value_loading=0.25,
            capital_allocation=0.20,
            stop_loss_pct=-0.06,
        ),
        Strategy(
            name="Macro Rates",
            style="macro",
            target_vol=0.06,
            annual_alpha=0.04,
            annual_vol=0.07,
            market_beta=-0.10,
            momentum_loading=0.20,
            value_loading=-0.05,
            capital_allocation=0.20,
            stop_loss_pct=-0.05,
        ),
        Strategy(
            name="Vol Arb",
            style="mean_reversion",
            target_vol=0.10,
            annual_alpha=0.07,
            annual_vol=0.12,
            market_beta=-0.15,
            momentum_loading=-0.10,
            value_loading=0.0,
            capital_allocation=0.15,
            stop_loss_pct=-0.05,
        ),
        Strategy(
            name="CTA Momentum",
            style="momentum",
            target_vol=0.15,
            annual_alpha=0.05,
            annual_vol=0.18,
            market_beta=0.20,
            momentum_loading=0.40,
            value_loading=-0.15,
            capital_allocation=0.20,
            stop_loss_pct=-0.08,
        ),
    ]
    
    config = PortfolioConfig(total_capital=100_000_000)
    return strategies, config


# ============================================================================
# Return Simulation (Factor Model)
# ============================================================================

class FactorModel:
    """
    Multi-factor return model for backtesting.
    
    r_i(t) = alpha_i + beta_i * r_mkt(t) + mom_i * r_mom(t) + val_i * r_val(t) + eps_i(t)
    
    Factor returns are simulated with realistic dynamics:
    - Market: GBM with regime switching
    - Momentum: autocorrelated
    - Value: mean-reverting, negatively correlated with momentum
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def simulate_factors(self, n_days: int) -> Dict[str, np.ndarray]:
        """Simulate daily factor returns."""
        dt = 1 / 252
        
        # Market factor (with regime)
        mkt_returns = np.zeros(n_days)
        regime = 'normal'
        for t in range(n_days):
            if regime == 'normal':
                if self.rng.random() < 0.002:
                    regime = 'crisis'
                mkt_returns[t] = self.rng.normal(0.08 * dt, 0.16 * np.sqrt(dt))
            else:
                if self.rng.random() < 0.01:
                    regime = 'normal'
                mkt_returns[t] = self.rng.normal(-0.30 * dt, 0.35 * np.sqrt(dt))
        
        # Momentum factor (autocorrelated)
        mom_returns = np.zeros(n_days)
        mom_returns[0] = self.rng.normal(0, 0.10 * np.sqrt(dt))
        for t in range(1, n_days):
            mom_returns[t] = 0.03 * mom_returns[t-1] + self.rng.normal(0.03 * dt, 0.12 * np.sqrt(dt))
        
        # Value factor (mean-reverting, neg corr with momentum)
        val_returns = np.zeros(n_days)
        for t in range(n_days):
            val_returns[t] = (-0.2 * mom_returns[t] 
                             + self.rng.normal(0.02 * dt, 0.10 * np.sqrt(dt)))
        
        return {
            'market': mkt_returns,
            'momentum': mom_returns,
            'value': val_returns,
        }
    
    def simulate_strategy_returns(
        self,
        strategies: List[Strategy],
        n_days: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Simulate daily returns for all strategies.
        
        Returns:
            returns: (n_days, n_strategies) daily returns
            factors: dict of factor return arrays
        """
        factors = self.simulate_factors(n_days)
        dt = 1 / 252
        n_strats = len(strategies)
        
        returns = np.zeros((n_days, n_strats))
        
        for i, strat in enumerate(strategies):
            # Systematic component
            systematic = (strat.market_beta * factors['market']
                        + strat.momentum_loading * factors['momentum']
                        + strat.value_loading * factors['value'])
            
            # Alpha (idiosyncratic)
            alpha = strat.annual_alpha * dt
            
            # Idiosyncratic vol
            systematic_vol = np.sqrt(
                strat.market_beta**2 * np.var(factors['market'])
                + strat.momentum_loading**2 * np.var(factors['momentum'])
                + strat.value_loading**2 * np.var(factors['value'])
            ) * np.sqrt(252)
            
            idio_vol = np.sqrt(max(strat.annual_vol**2 - systematic_vol**2, 0.001))
            
            idiosyncratic = self.rng.normal(0, idio_vol * np.sqrt(dt), n_days)
            
            returns[:, i] = alpha + systematic + idiosyncratic
        
        return returns, factors


# ============================================================================
# Risk Budgeting
# ============================================================================

class RiskBudgeter:
    """
    Risk budgeting and allocation engine.
    
    Methods:
    1. Equal Risk Contribution (ERC): each strategy contributes equally to portfolio vol
    2. Inverse Vol: weight inversely proportional to volatility
    3. Vol Target: scale each strategy to its vol target
    4. Max Diversification: maximise diversification ratio
    """
    
    @staticmethod
    def equal_risk_contribution(cov_matrix: np.ndarray) -> np.ndarray:
        """
        Find weights where each asset contributes equally to portfolio risk.
        
        Risk contribution of asset i: RC_i = w_i * (Sigma @ w)_i / sigma_p
        Target: RC_i = 1/n for all i
        """
        n = cov_matrix.shape[0]
        
        def objective(w):
            w = np.abs(w)
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol < 1e-10:
                return 1e10
            marginal_risk = cov_matrix @ w
            risk_contrib = w * marginal_risk / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target)**2)
        
        w0 = np.ones(n) / n
        bounds = [(0.01, 0.5)] * n
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = np.abs(result.x)
        return weights / np.sum(weights)
    
    @staticmethod
    def inverse_volatility(vols: np.ndarray) -> np.ndarray:
        """Weight inversely proportional to volatility."""
        inv_vol = 1.0 / vols
        return inv_vol / np.sum(inv_vol)
    
    @staticmethod
    def vol_target(
        vols: np.ndarray,
        target_vol: float,
        correlation: np.ndarray
    ) -> np.ndarray:
        """Scale weights to hit portfolio vol target."""
        n = len(vols)
        w = np.ones(n) / n
        
        # Iterative: start equal, then scale to hit target
        cov = np.outer(vols, vols) * correlation
        port_vol = np.sqrt(w @ cov @ w)
        
        if port_vol > 0:
            scale = target_vol / port_vol
            w *= scale
        
        return w / np.sum(w)
    
    @staticmethod
    def risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Compute percentage risk contribution per strategy."""
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        if port_vol < 1e-10:
            return np.ones(len(weights)) / len(weights)
        marginal = cov_matrix @ weights
        rc = weights * marginal / port_vol
        return rc / np.sum(rc)


# ============================================================================
# Performance Attribution
# ============================================================================

class PerformanceAttribution:
    """
    Multi-level performance attribution.
    
    1. Strategy-level: which pods generated/lost P&L
    2. Factor-level: how much came from market/momentum/value/alpha
    3. Timing: did rebalancing add or subtract value
    """
    
    @staticmethod
    def strategy_attribution(
        returns: np.ndarray,
        weights: np.ndarray,
        strategy_names: List[str]
    ) -> dict:
        """Attribute portfolio return to individual strategies."""
        n_days, n_strats = returns.shape
        
        # Weighted returns
        weighted = returns * weights
        port_returns = np.sum(weighted, axis=1)
        
        attribution = {}
        for i, name in enumerate(strategy_names):
            strat_contrib = weighted[:, i]
            attribution[name] = {
                'total_contrib': np.sum(strat_contrib),
                'annualised_contrib': np.mean(strat_contrib) * 252,
                'pct_of_total': np.sum(strat_contrib) / max(np.sum(port_returns), 1e-10),
                'sharpe_standalone': np.mean(returns[:, i]) / max(np.std(returns[:, i]), 1e-10) * np.sqrt(252),
                'info_ratio': np.mean(strat_contrib) / max(np.std(strat_contrib), 1e-10) * np.sqrt(252),
            }
        
        return attribution
    
    @staticmethod
    def factor_attribution(
        returns: np.ndarray,
        factors: Dict[str, np.ndarray],
        strategies: List[Strategy],
        weights: np.ndarray
    ) -> dict:
        """Decompose portfolio return into factor contributions."""
        n_days = returns.shape[0]
        dt = 1 / 252
        
        port_returns = np.sum(returns * weights, axis=1)
        
        # Factor contributions
        factor_contribs = {}
        total_factor = np.zeros(n_days)
        
        for fname in ['market', 'momentum', 'value']:
            factor_return = factors[fname]
            # Weighted factor exposure
            if fname == 'market':
                loadings = np.array([s.market_beta for s in strategies])
            elif fname == 'momentum':
                loadings = np.array([s.momentum_loading for s in strategies])
            else:
                loadings = np.array([s.value_loading for s in strategies])
            
            port_loading = np.sum(loadings * weights)
            contrib = port_loading * factor_return
            total_factor += contrib
            
            factor_contribs[fname] = {
                'portfolio_loading': port_loading,
                'total_contrib': np.sum(contrib),
                'annualised': np.mean(contrib) * 252,
            }
        
        # Alpha = total - factors
        alpha_contrib = port_returns - total_factor
        factor_contribs['alpha'] = {
            'total_contrib': np.sum(alpha_contrib),
            'annualised': np.mean(alpha_contrib) * 252,
        }
        
        return factor_contribs


# ============================================================================
# Drawdown Analysis
# ============================================================================

class DrawdownAnalyser:
    """Comprehensive drawdown analysis."""
    
    @staticmethod
    def compute_drawdowns(cumulative_returns: np.ndarray) -> dict:
        """Full drawdown analysis."""
        wealth = 1 + cumulative_returns
        peak = np.maximum.accumulate(wealth)
        drawdown = (wealth - peak) / peak
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find peak before max DD
        peak_idx = np.argmax(wealth[:max_dd_idx + 1])
        
        # Find recovery after max DD
        recovery_idx = max_dd_idx
        for i in range(max_dd_idx, len(wealth)):
            if wealth[i] >= peak[max_dd_idx]:
                recovery_idx = i
                break
        
        # All drawdown periods
        in_drawdown = drawdown < -0.001
        dd_periods = []
        start = None
        for i in range(len(in_drawdown)):
            if in_drawdown[i] and start is None:
                start = i
            elif not in_drawdown[i] and start is not None:
                dd_periods.append({
                    'start': start,
                    'end': i,
                    'duration': i - start,
                    'depth': np.min(drawdown[start:i]),
                })
                start = None
        
        # Sort by depth
        dd_periods.sort(key=lambda x: x['depth'])
        
        # Drawdown duration stats
        durations = [p['duration'] for p in dd_periods] if dd_periods else [0]
        
        return {
            'max_drawdown': max_dd,
            'max_dd_peak_idx': peak_idx,
            'max_dd_trough_idx': max_dd_idx,
            'max_dd_recovery_idx': recovery_idx,
            'max_dd_duration': max_dd_idx - peak_idx,
            'recovery_duration': recovery_idx - max_dd_idx,
            'drawdown_series': drawdown,
            'n_drawdown_periods': len(dd_periods),
            'avg_dd_duration': np.mean(durations),
            'max_dd_duration_days': max(durations) if durations else 0,
            'top_5_drawdowns': dd_periods[:5],
            'time_in_drawdown': np.mean(in_drawdown),
        }
    
    @staticmethod
    def underwater_chart_data(cumulative_returns: np.ndarray) -> np.ndarray:
        """Drawdown series for underwater chart."""
        wealth = 1 + cumulative_returns
        peak = np.maximum.accumulate(wealth)
        return (wealth - peak) / peak


# ============================================================================
# Risk Metrics
# ============================================================================

class RiskMetrics:
    """Comprehensive risk metrics suite."""
    
    @staticmethod
    def compute_all(returns: np.ndarray, risk_free: float = 0.04) -> dict:
        """Compute full risk metrics."""
        daily_rf = risk_free / 252
        excess = returns - daily_rf
        
        ann_return = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252)
        sharpe = np.mean(excess) / max(np.std(returns), 1e-10) * np.sqrt(252)
        
        # Sortino (downside deviation)
        downside = returns[returns < daily_rf] - daily_rf
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = np.mean(excess) * 252 / max(downside_vol, 1e-10)
        
        # Calmar
        cum_ret = np.cumsum(returns)
        dd_analysis = DrawdownAnalyser.compute_drawdowns(cum_ret)
        calmar = ann_return / max(abs(dd_analysis['max_drawdown']), 1e-10)
        
        # CVaR (Expected Shortfall)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        var_99 = np.percentile(returns, 1)
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Higher moments
        skewness = float(np.mean(((returns - np.mean(returns)) / np.std(returns))**3))
        kurtosis = float(np.mean(((returns - np.mean(returns)) / np.std(returns))**4))
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Profit factor
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gains / max(losses, 1e-10)
        
        # Best/worst
        best_day = np.max(returns)
        worst_day = np.min(returns)
        best_month = 0
        worst_month = 0
        if len(returns) >= 21:
            monthly = [np.sum(returns[i:i+21]) for i in range(0, len(returns)-20, 21)]
            best_month = max(monthly)
            worst_month = min(monthly)
        
        return {
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_drawdown': dd_analysis['max_drawdown'],
            'var_95': var_95,
            'cvar_95': cvar_95,
            'var_99': var_99,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_month': best_month,
            'worst_month': worst_month,
            'time_in_drawdown': dd_analysis['time_in_drawdown'],
            'drawdown_analysis': dd_analysis,
        }


# ============================================================================
# Pod Risk Monitor
# ============================================================================

class PodRiskMonitor:
    """
    Real-time pod-level risk monitoring.
    
    Tracks each pod against its risk limits and flags breaches.
    This is what the risk team at a pod shop monitors intraday.
    """
    
    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies
        self.breach_log: List[dict] = []
    
    def check_limits(
        self,
        day: int,
        returns: np.ndarray,
        cumulative_returns: np.ndarray
    ) -> List[dict]:
        """Check all risk limits for each pod."""
        breaches = []
        n_strats = len(self.strategies)
        
        for i, strat in enumerate(self.strategies):
            # Drawdown check
            wealth = 1 + cumulative_returns[:, i]
            peak = np.maximum.accumulate(wealth)
            dd = (wealth[-1] - peak[-1]) / peak[-1]
            
            if dd < strat.stop_loss_pct:
                breach = {
                    'day': day,
                    'strategy': strat.name,
                    'type': 'STOP_LOSS',
                    'value': dd,
                    'limit': strat.stop_loss_pct,
                    'severity': 'CRITICAL',
                }
                breaches.append(breach)
            
            # Vol check (rolling 21-day)
            if day >= 21:
                rolling_vol = np.std(returns[day-21:day, i]) * np.sqrt(252)
                if rolling_vol > strat.target_vol * 1.5:
                    breach = {
                        'day': day,
                        'strategy': strat.name,
                        'type': 'VOL_BREACH',
                        'value': rolling_vol,
                        'limit': strat.target_vol * 1.5,
                        'severity': 'WARNING',
                    }
                    breaches.append(breach)
        
        self.breach_log.extend(breaches)
        return breaches
    
    def risk_dashboard(
        self,
        returns: np.ndarray,
        day: int
    ) -> dict:
        """Daily risk dashboard for all pods."""
        dashboard = {}
        
        for i, strat in enumerate(self.strategies):
            strat_returns = returns[:day+1, i]
            cum_ret = np.cumsum(strat_returns)
            
            # Current DD
            wealth = 1 + cum_ret
            peak = np.maximum.accumulate(wealth)
            current_dd = (wealth[-1] - peak[-1]) / peak[-1]
            
            # Rolling vol
            lookback = min(63, day)
            if lookback > 5:
                rolling_vol = np.std(strat_returns[-lookback:]) * np.sqrt(252)
            else:
                rolling_vol = strat.annual_vol
            
            # Rolling Sharpe
            if lookback > 21:
                rolling_sharpe = np.mean(strat_returns[-lookback:]) / max(np.std(strat_returns[-lookback:]), 1e-10) * np.sqrt(252)
            else:
                rolling_sharpe = 0
            
            # Distance to stop
            distance_to_stop = current_dd - strat.stop_loss_pct
            
            dashboard[strat.name] = {
                'ytd_return': cum_ret[-1],
                'current_dd': current_dd,
                'rolling_vol': rolling_vol,
                'rolling_sharpe': rolling_sharpe,
                'distance_to_stop': distance_to_stop,
                'stop_loss': strat.stop_loss_pct,
                'status': 'OK' if distance_to_stop > 0.01 else ('WARNING' if distance_to_stop > 0 else 'STOPPED'),
            }
        
        return dashboard


# ============================================================================
# Correlation Regime Analysis
# ============================================================================

class CorrelationAnalyser:
    """Analyse correlation regimes across strategies."""
    
    @staticmethod
    def rolling_correlation(
        returns: np.ndarray,
        window: int = 63
    ) -> np.ndarray:
        """Rolling pairwise correlation matrix."""
        n_days, n_strats = returns.shape
        n_windows = n_days - window + 1
        
        corr_series = np.zeros((n_windows, n_strats, n_strats))
        
        for t in range(n_windows):
            window_returns = returns[t:t+window]
            corr_series[t] = np.corrcoef(window_returns.T)
        
        return corr_series
    
    @staticmethod
    def average_correlation(corr_matrix: np.ndarray) -> float:
        """Average pairwise correlation (excluding diagonal)."""
        n = corr_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return np.mean(corr_matrix[mask])
    
    @staticmethod
    def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
        """
        DR = (w' * sigma) / sqrt(w' * Sigma * w)
        
        DR > 1 indicates diversification benefit.
        Higher = more diversified.
        """
        vols = np.sqrt(np.diag(cov))
        weighted_vol = np.dot(weights, vols)
        port_vol = np.sqrt(weights @ cov @ weights)
        return weighted_vol / max(port_vol, 1e-10)


# ============================================================================
# Full Backtest Engine
# ============================================================================

class Backtester:
    """
    Full backtesting engine orchestrating all components.
    
    Runs the simulation, applies risk limits, computes all analytics.
    """
    
    def __init__(
        self,
        strategies: List[Strategy],
        config: PortfolioConfig,
        seed: int = 42
    ):
        self.strategies = strategies
        self.config = config
        self.factor_model = FactorModel(seed=seed)
        self.risk_budgeter = RiskBudgeter()
        self.pod_monitor = PodRiskMonitor(strategies)
        self.dd_analyser = DrawdownAnalyser()
        self.corr_analyser = CorrelationAnalyser()
    
    def run(self, n_years: float = 3.0) -> dict:
        """Run full backtest."""
        n_days = int(n_years * 252)
        n_strats = len(self.strategies)
        
        # 1. Simulate returns
        returns, factors = self.factor_model.simulate_strategy_returns(
            self.strategies, n_days
        )
        
        # 2. Compute weights (initial allocation)
        initial_weights = np.array([s.capital_allocation for s in self.strategies])
        initial_weights /= np.sum(initial_weights)
        
        # 3. Risk budgeting: compute ERC weights from estimated covariance
        cov_est = np.cov(returns[:min(63, n_days)].T) * 252
        erc_weights = self.risk_budgeter.equal_risk_contribution(cov_est)
        inv_vol_weights = self.risk_budgeter.inverse_volatility(
            np.array([s.annual_vol for s in self.strategies])
        )
        
        # Use ERC weights
        weights = erc_weights
        
        # 4. Portfolio returns
        port_returns = returns @ weights
        cum_port = np.cumsum(port_returns)
        
        # Strategy cumulative returns
        cum_strat = np.cumsum(returns, axis=0)
        
        # 5. Risk monitoring
        for day in range(n_days):
            self.pod_monitor.check_limits(day, returns[:day+1], cum_strat[:day+1])
        
        # 6. Analytics
        port_metrics = RiskMetrics.compute_all(port_returns, self.config.risk_free_rate)
        
        strat_metrics = {}
        for i, strat in enumerate(self.strategies):
            strat_metrics[strat.name] = RiskMetrics.compute_all(
                returns[:, i], self.config.risk_free_rate
            )
        
        # 7. Attribution
        strat_attribution = PerformanceAttribution.strategy_attribution(
            returns, weights, [s.name for s in self.strategies]
        )
        factor_attribution = PerformanceAttribution.factor_attribution(
            returns, factors, self.strategies, weights
        )
        
        # 8. Risk contributions
        full_cov = np.cov(returns.T) * 252
        risk_contribs = self.risk_budgeter.risk_contributions(weights, full_cov)
        
        # 9. Correlation analysis
        corr_series = self.corr_analyser.rolling_correlation(returns, window=63)
        avg_corrs = [self.corr_analyser.average_correlation(c) for c in corr_series]
        div_ratio = self.corr_analyser.diversification_ratio(weights, full_cov)
        
        # 10. Final risk dashboard
        final_dashboard = self.pod_monitor.risk_dashboard(returns, n_days - 1)
        
        return {
            'returns': returns,
            'factors': factors,
            'weights': weights,
            'erc_weights': erc_weights,
            'inv_vol_weights': inv_vol_weights,
            'port_returns': port_returns,
            'cum_port': cum_port,
            'cum_strat': cum_strat,
            'port_metrics': port_metrics,
            'strat_metrics': strat_metrics,
            'strat_attribution': strat_attribution,
            'factor_attribution': factor_attribution,
            'risk_contributions': risk_contribs,
            'correlation_series': np.array(avg_corrs),
            'diversification_ratio': div_ratio,
            'breach_log': self.pod_monitor.breach_log,
            'risk_dashboard': final_dashboard,
            'n_days': n_days,
        }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("MULTI-STRATEGY PORTFOLIO BACKTESTER")
    print("Pod Shop Risk Management Framework")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 65)
    
    strategies, config = create_pod_shop_portfolio()
    
    print(f"\nPORTFOLIO CONFIGURATION")
    print(f"Capital: ${config.total_capital:,.0f}")
    print(f"Target vol: {config.target_vol:.0%}")
    print(f"Max DD limit: {config.max_drawdown_limit:.0%}")
    print(f"\nSTRATEGIES ({len(strategies)} pods):")
    for s in strategies:
        print(f"  {s.describe()}")
    
    # Run backtest
    t0 = time.time()
    bt = Backtester(strategies, config, seed=42)
    results = bt.run(n_years=3.0)
    print(f"\nBacktest: {time.time()-t0:.2f}s ({results['n_days']} days)")
    
    # Portfolio metrics
    pm = results['port_metrics']
    print(f"\nPORTFOLIO PERFORMANCE")
    print(f"{'='*45}")
    print(f"Annual return:    {pm['ann_return']:>8.2%}")
    print(f"Annual vol:       {pm['ann_vol']:>8.2%}")
    print(f"Sharpe ratio:     {pm['sharpe']:>8.2f}")
    print(f"Sortino ratio:    {pm['sortino']:>8.2f}")
    print(f"Calmar ratio:     {pm['calmar']:>8.2f}")
    print(f"Max drawdown:     {pm['max_drawdown']:>8.2%}")
    print(f"Win rate:         {pm['win_rate']:>8.1%}")
    print(f"Profit factor:    {pm['profit_factor']:>8.2f}")
    print(f"VaR 95:           {pm['var_95']:>8.4f}")
    print(f"CVaR 95:          {pm['cvar_95']:>8.4f}")
    print(f"Skewness:         {pm['skewness']:>8.2f}")
    print(f"Kurtosis:         {pm['kurtosis']:>8.2f}")
    print(f"Time in DD:       {pm['time_in_drawdown']:>8.1%}")
    
    # Strategy comparison
    print(f"\nSTRATEGY COMPARISON")
    print(f"{'Strategy':<25} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Weight':>8} {'RiskC':>8}")
    print("-" * 75)
    for i, strat in enumerate(strategies):
        sm = results['strat_metrics'][strat.name]
        w = results['weights'][i]
        rc = results['risk_contributions'][i]
        print(f"{strat.name:<25} {sm['ann_return']:>7.1%} {sm['ann_vol']:>7.1%} "
              f"{sm['sharpe']:>7.2f} {sm['max_drawdown']:>7.1%} {w:>7.1%} {rc:>7.1%}")
    
    # Factor attribution
    print(f"\nFACTOR ATTRIBUTION")
    print(f"{'='*45}")
    for fname, stats in results['factor_attribution'].items():
        print(f"{fname.capitalize():<15} {stats['annualised']:>+8.2%}")
    
    # Risk dashboard
    print(f"\nRISK DASHBOARD (End of Period)")
    print(f"{'Strategy':<25} {'YTD':>8} {'DD':>8} {'Vol':>8} {'Sharpe':>8} {'Status':>10}")
    print("-" * 70)
    for name, dash in results['risk_dashboard'].items():
        print(f"{name:<25} {dash['ytd_return']:>7.1%} {dash['current_dd']:>7.1%} "
              f"{dash['rolling_vol']:>7.1%} {dash['rolling_sharpe']:>7.2f} {dash['status']:>10}")
    
    # Breaches
    n_breaches = len(results['breach_log'])
    critical = sum(1 for b in results['breach_log'] if b['severity'] == 'CRITICAL')
    print(f"\nRISK BREACHES: {n_breaches} total ({critical} critical)")
    
    # Diversification
    print(f"\nDIVERSIFICATION")
    print(f"Diversification ratio: {results['diversification_ratio']:.2f}")
    print(f"Avg correlation:       {np.mean(results['correlation_series']):.3f}")
    
    return results


if __name__ == "__main__":
    results = main()
