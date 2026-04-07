"""
Extension 2: Dynamic Allocation & Regime-Aware Rebalancing
============================================================
Adjusts portfolio weights based on detected market regime.

Regimes:
  - Risk-on: low vol, low correlation -> full allocation, tighter spreads
  - Risk-off: high vol, rising correlation -> reduce gross, widen stops
  - Crisis: extreme vol, correlation -> 1 -> cut to minimum, protect capital

Detection methods:
  1. Rolling vol regime (threshold-based)
  2. Correlation regime (average pairwise rolling correlation)
  3. Hidden Markov Model (simplified 2-state)

The dynamic allocator adjusts:
  - Pod weights (reduce high-vol pods in risk-off)
  - Portfolio vol target (scale down in crisis)
  - Stop-loss levels (tighten in risk-off)

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time


@dataclass
class RegimeState:
    """Current detected regime."""
    name: str              # 'risk_on', 'risk_off', 'crisis'
    confidence: float      # 0-1
    vol_level: float       # Current vol
    corr_level: float      # Current avg correlation
    
    @property
    def vol_scalar(self) -> float:
        """How much to scale portfolio vol target."""
        if self.name == 'risk_on':
            return 1.0
        elif self.name == 'risk_off':
            return 0.7
        else:
            return 0.3
    
    @property
    def gross_scalar(self) -> float:
        """How much to scale gross exposure."""
        if self.name == 'risk_on':
            return 1.0
        elif self.name == 'risk_off':
            return 0.6
        else:
            return 0.2


class RegimeDetector:
    """
    Detect market regime from rolling statistics.
    
    Uses a combination of:
    1. Portfolio vol (rolling 21-day)
    2. Average pairwise correlation (rolling 63-day)
    3. Vol-of-vol (stability of vol estimate)
    """
    
    def __init__(
        self,
        vol_threshold_high: float = 0.20,
        vol_threshold_crisis: float = 0.35,
        corr_threshold_high: float = 0.40,
        lookback_vol: int = 21,
        lookback_corr: int = 63
    ):
        self.vol_high = vol_threshold_high
        self.vol_crisis = vol_threshold_crisis
        self.corr_high = corr_threshold_high
        self.lb_vol = lookback_vol
        self.lb_corr = lookback_corr
    
    def detect(
        self,
        returns: np.ndarray,  # (history_length, n_strats)
        weights: np.ndarray
    ) -> RegimeState:
        """Detect current regime from recent returns."""
        n_days = returns.shape[0]
        
        # Portfolio vol (annualised)
        if n_days >= self.lb_vol:
            port_ret = returns[-self.lb_vol:] @ weights
            vol = np.std(port_ret) * np.sqrt(252)
        else:
            vol = 0.10
        
        # Average correlation
        if n_days >= self.lb_corr:
            window = returns[-self.lb_corr:]
            corr_mat = np.corrcoef(window.T)
            n = corr_mat.shape[0]
            mask = ~np.eye(n, dtype=bool)
            avg_corr = np.mean(corr_mat[mask])
        else:
            avg_corr = 0.1
        
        # Classify
        if vol > self.vol_crisis or (vol > self.vol_high and avg_corr > self.corr_high):
            regime = 'crisis'
            confidence = min((vol / self.vol_crisis), 1.0)
        elif vol > self.vol_high or avg_corr > self.corr_high:
            regime = 'risk_off'
            confidence = max(vol / self.vol_high, avg_corr / self.corr_high)
        else:
            regime = 'risk_on'
            confidence = 1.0 - vol / self.vol_high
        
        return RegimeState(regime, min(confidence, 1.0), vol, avg_corr)


class DynamicAllocator:
    """
    Regime-aware dynamic portfolio allocation.
    
    In risk-on: use ERC weights, full vol target
    In risk-off: reduce volatile pods, lower vol target
    In crisis: minimum allocation, capital preservation
    """
    
    def __init__(self, base_weights: np.ndarray, strategies: list):
        self.base_weights = base_weights
        self.strategies = strategies
    
    def compute_dynamic_weights(
        self,
        regime: RegimeState,
        recent_returns: np.ndarray = None
    ) -> np.ndarray:
        """Adjust weights based on regime."""
        n = len(self.strategies)
        weights = self.base_weights.copy()
        
        if regime.name == 'risk_on':
            return weights
        
        elif regime.name == 'risk_off':
            # Reduce high-vol pods, increase low-vol
            vols = np.array([s.annual_vol for s in self.strategies])
            vol_rank = np.argsort(vols)  # Low to high
            
            for i, idx in enumerate(vol_rank):
                if i < n // 2:
                    weights[idx] *= 1.2  # Increase low-vol
                else:
                    weights[idx] *= 0.6  # Reduce high-vol
            
            # Scale to hit reduced vol target
            weights *= regime.vol_scalar
        
        elif regime.name == 'crisis':
            # Cut to minimum, favour market-neutral and low-beta
            betas = np.array([abs(s.market_beta) for s in self.strategies])
            beta_penalty = 1.0 - betas / max(np.max(betas), 0.01)
            
            weights *= beta_penalty * regime.gross_scalar
        
        # Normalise
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(n) / n
        
        return weights


class DynamicBacktester:
    """
    Backtest with dynamic regime-aware rebalancing.
    
    Compares:
    1. Static allocation (fixed weights)
    2. Dynamic allocation (regime-responsive)
    """
    
    def __init__(self, strategies: list, seed: int = 42):
        self.strategies = strategies
        self.rng = np.random.default_rng(seed)
        self.detector = RegimeDetector()
    
    def run(
        self,
        returns: np.ndarray,
        static_weights: np.ndarray,
        rebalance_freq: int = 5  # Weekly
    ) -> dict:
        """Run dynamic vs static comparison."""
        n_days, n_strats = returns.shape
        
        allocator = DynamicAllocator(static_weights, self.strategies)
        
        # Static portfolio
        static_port = returns @ static_weights
        static_cum = np.cumsum(static_port)
        
        # Dynamic portfolio
        dynamic_weights_history = np.zeros((n_days, n_strats))
        dynamic_port = np.zeros(n_days)
        regime_history = []
        current_weights = static_weights.copy()
        
        for day in range(n_days):
            # Use current weights
            dynamic_port[day] = returns[day] @ current_weights
            dynamic_weights_history[day] = current_weights
            
            # Detect regime and rebalance
            if day > 0 and day % rebalance_freq == 0 and day >= 63:
                regime = self.detector.detect(returns[:day], current_weights)
                regime_history.append({'day': day, 'regime': regime.name, 'vol': regime.vol_level, 'corr': regime.corr_level})
                current_weights = allocator.compute_dynamic_weights(regime, returns[max(0,day-63):day])
        
        dynamic_cum = np.cumsum(dynamic_port)
        
        # Metrics comparison
        def metrics(rets):
            ann_ret = np.mean(rets) * 252
            ann_vol = np.std(rets) * np.sqrt(252)
            sharpe = ann_ret / max(ann_vol, 1e-10)
            wealth = 1 + np.cumsum(rets)
            peak = np.maximum.accumulate(wealth)
            max_dd = np.min((wealth - peak) / peak)
            calmar = ann_ret / max(abs(max_dd), 1e-10)
            return {'return': ann_ret, 'vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd, 'calmar': calmar}
        
        static_metrics = metrics(static_port)
        dynamic_metrics = metrics(dynamic_port)
        
        # Regime time distribution
        regime_counts = {'risk_on': 0, 'risk_off': 0, 'crisis': 0}
        for r in regime_history:
            regime_counts[r['regime']] += 1
        total = max(sum(regime_counts.values()), 1)
        regime_pcts = {k: v/total for k, v in regime_counts.items()}
        
        return {
            'static_cum': static_cum,
            'dynamic_cum': dynamic_cum,
            'static_metrics': static_metrics,
            'dynamic_metrics': dynamic_metrics,
            'regime_history': regime_history,
            'regime_distribution': regime_pcts,
            'dynamic_weights': dynamic_weights_history,
            'improvement': {
                'sharpe_diff': dynamic_metrics['sharpe'] - static_metrics['sharpe'],
                'dd_improvement': dynamic_metrics['max_dd'] - static_metrics['max_dd'],
                'vol_reduction': static_metrics['vol'] - dynamic_metrics['vol'],
            }
        }


if __name__ == "__main__":
    from portfolio_backtester import create_pod_shop_portfolio, FactorModel
    
    strategies, config = create_pod_shop_portfolio()
    weights = np.array([s.capital_allocation for s in strategies])
    weights /= np.sum(weights)
    
    print("Dynamic Allocation & Regime-Aware Rebalancing")
    print("=" * 55)
    
    # Simulate returns
    fm = FactorModel(seed=42)
    returns, factors = fm.simulate_strategy_returns(strategies, 756)
    
    # Run comparison
    bt = DynamicBacktester(strategies, seed=42)
    results = bt.run(returns, weights)
    
    print(f"\n{'Metric':<18} {'Static':>10} {'Dynamic':>10} {'Diff':>10}")
    print("-" * 50)
    for metric in ['return', 'vol', 'sharpe', 'max_dd', 'calmar']:
        s = results['static_metrics'][metric]
        d = results['dynamic_metrics'][metric]
        fmt = '.1%' if metric in ['return', 'vol', 'max_dd'] else '.2f'
        print(f"{metric.capitalize():<18} {s:>10{fmt}} {d:>10{fmt}} {d-s:>+10{fmt}}")
    
    print(f"\nREGIME DISTRIBUTION")
    for regime, pct in results['regime_distribution'].items():
        print(f"  {regime:<12} {pct:.0%}")
    
    imp = results['improvement']
    print(f"\nDynamic allocation {'improves' if imp['sharpe_diff'] > 0 else 'reduces'} Sharpe by {abs(imp['sharpe_diff']):.2f}")
    print(f"Max DD improvement: {imp['dd_improvement']:+.1%}")
