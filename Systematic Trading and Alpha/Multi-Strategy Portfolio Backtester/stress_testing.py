"""
Extension 1: Stress Testing & Scenario Engine
================================================
Replay historical crisis scenarios and compute P&L impact per pod.

Scenarios:
  - GFC 2008: equity -55%, vol +300%, credit spreads +500bps, correlations -> 1
  - COVID Mar 2020: equity -34%, vol +400%, rates -150bps, recovery in 5 months
  - 2022 Rates Shock: rates +300bps, equity -25%, growth/momentum crash
  - EM Crisis: EM equities -40%, FX -20%, rates +200bps, flight to quality
  - Vol Shock (Volmageddon 2018): short vol strategies -90%, VIX spike

Each scenario defines shocks to factors (market, momentum, value) and
correlation overrides. The engine applies these to the portfolio and
reports pod-level P&L, worst pod, portfolio DD, and recovery time.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import time


@dataclass
class CrisisScenario:
    """Historical crisis scenario definition."""
    name: str
    description: str
    duration_days: int
    
    # Factor shocks (cumulative over the crisis period)
    market_shock: float = 0.0          # Equity market return
    momentum_shock: float = 0.0        # Momentum factor return
    value_shock: float = 0.0           # Value factor return
    
    # Vol multiplier during crisis
    vol_multiplier: float = 1.0
    
    # Correlation override (all pairwise -> this level)
    crisis_correlation: float = 0.0    # 0 = no override
    
    # Recovery
    recovery_days: int = 0             # Days to recover to pre-crisis level
    recovery_strength: float = 0.5     # How much of the loss is recovered


def create_crisis_scenarios() -> List[CrisisScenario]:
    """Create suite of historical crisis scenarios."""
    return [
        CrisisScenario(
            name="GFC 2008",
            description="Global Financial Crisis: Lehman collapse, credit freeze, equity crash",
            duration_days=126,  # ~6 months
            market_shock=-0.55,
            momentum_shock=-0.30,
            value_shock=-0.20,
            vol_multiplier=3.5,
            crisis_correlation=0.80,
            recovery_days=252,
            recovery_strength=0.60,
        ),
        CrisisScenario(
            name="COVID Mar 2020",
            description="Pandemic shock: fastest bear market in history, V-shaped recovery",
            duration_days=23,  # ~1 month
            market_shock=-0.34,
            momentum_shock=-0.25,
            value_shock=-0.15,
            vol_multiplier=4.0,
            crisis_correlation=0.85,
            recovery_days=105,
            recovery_strength=0.90,
        ),
        CrisisScenario(
            name="2022 Rates Shock",
            description="Fed hiking cycle: duration sell-off, growth-to-value rotation",
            duration_days=189,  # ~9 months
            market_shock=-0.25,
            momentum_shock=-0.35,
            value_shock=0.15,
            vol_multiplier=1.8,
            crisis_correlation=0.50,
            recovery_days=126,
            recovery_strength=0.40,
        ),
        CrisisScenario(
            name="EM Crisis",
            description="Emerging market contagion: currency crash, flight to quality",
            duration_days=63,
            market_shock=-0.20,
            momentum_shock=-0.15,
            value_shock=-0.25,
            vol_multiplier=2.5,
            crisis_correlation=0.65,
            recovery_days=189,
            recovery_strength=0.50,
        ),
        CrisisScenario(
            name="Volmageddon 2018",
            description="VIX spike, short-vol strategies collapse, XIV liquidation",
            duration_days=5,
            market_shock=-0.10,
            momentum_shock=-0.05,
            value_shock=0.02,
            vol_multiplier=5.0,
            crisis_correlation=0.70,
            recovery_days=42,
            recovery_strength=0.70,
        ),
        CrisisScenario(
            name="Quant Quake 2007",
            description="Factor crowding unwind: stat arb strategies simultaneously deleveraging",
            duration_days=10,
            market_shock=-0.05,
            momentum_shock=-0.40,
            value_shock=-0.30,
            vol_multiplier=3.0,
            crisis_correlation=0.75,
            recovery_days=63,
            recovery_strength=0.30,
        ),
    ]


class StressTestEngine:
    """
    Apply crisis scenarios to a multi-strategy portfolio.
    
    For each scenario:
    1. Generate stressed factor returns over the crisis period
    2. Apply factor loadings per strategy
    3. Override correlations to crisis levels
    4. Compute pod-level and portfolio-level P&L
    5. Simulate recovery phase
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def apply_scenario(
        self,
        scenario: CrisisScenario,
        strategies: list,
        weights: np.ndarray
    ) -> dict:
        """Apply a single crisis scenario."""
        n_days = scenario.duration_days
        n_strats = len(strategies)
        dt = 1 / 252
        
        # Daily factor shocks (distribute cumulative shock over crisis period)
        daily_mkt = scenario.market_shock / n_days
        daily_mom = scenario.momentum_shock / n_days
        daily_val = scenario.value_shock / n_days
        
        # Generate stressed returns with higher vol and correlation
        vol_mult = scenario.vol_multiplier
        
        # Crisis correlation matrix
        if scenario.crisis_correlation > 0:
            corr = np.full((n_strats, n_strats), scenario.crisis_correlation)
            np.fill_diagonal(corr, 1.0)
            L = np.linalg.cholesky(corr)
        else:
            L = np.eye(n_strats)
        
        strat_returns = np.zeros((n_days, n_strats))
        
        for day in range(n_days):
            # Correlated noise
            eps = self.rng.standard_normal(n_strats)
            corr_eps = L @ eps
            
            for i, strat in enumerate(strategies):
                # Factor contribution
                factor_ret = (strat.market_beta * daily_mkt
                            + strat.momentum_loading * daily_mom
                            + strat.value_loading * daily_val)
                
                # Stressed idiosyncratic
                idio_vol = strat.annual_vol * vol_mult * np.sqrt(dt)
                idio = idio_vol * corr_eps[i]
                
                # No alpha during crisis
                strat_returns[day, i] = factor_ret + idio
        
        # Portfolio returns
        port_returns = strat_returns @ weights
        
        # Cumulative
        cum_strat = np.cumsum(strat_returns, axis=0)
        cum_port = np.cumsum(port_returns)
        
        # Recovery phase
        recovery_returns = np.zeros(scenario.recovery_days)
        if scenario.recovery_days > 0:
            total_loss = cum_port[-1]
            daily_recovery = -total_loss * scenario.recovery_strength / scenario.recovery_days
            for day in range(scenario.recovery_days):
                recovery_returns[day] = daily_recovery + self.rng.normal(0, abs(daily_recovery) * 0.5)
        
        cum_recovery = np.cumsum(recovery_returns) + cum_port[-1]
        
        # Pod-level analysis
        pod_results = {}
        for i, strat in enumerate(strategies):
            pod_results[strat.name] = {
                'total_return': cum_strat[-1, i],
                'max_drawdown': np.min(cum_strat[:, i]),
                'worst_day': np.min(strat_returns[:, i]),
                'contribution': cum_strat[-1, i] * weights[i],
            }
        
        # Find worst pod
        worst_pod = min(pod_results.items(), key=lambda x: x[1]['total_return'])
        
        return {
            'scenario': scenario.name,
            'port_total_return': cum_port[-1],
            'port_max_drawdown': np.min(cum_port),
            'port_worst_day': np.min(port_returns),
            'pod_results': pod_results,
            'worst_pod': worst_pod[0],
            'worst_pod_return': worst_pod[1]['total_return'],
            'cum_port': cum_port,
            'cum_strat': cum_strat,
            'recovery_final': cum_recovery[-1] if len(cum_recovery) > 0 else cum_port[-1],
            'full_path': np.concatenate([cum_port, cum_recovery]),
        }
    
    def run_all_scenarios(
        self,
        strategies: list,
        weights: np.ndarray
    ) -> Dict[str, dict]:
        """Run all crisis scenarios."""
        scenarios = create_crisis_scenarios()
        results = {}
        
        for scenario in scenarios:
            results[scenario.name] = self.apply_scenario(scenario, strategies, weights)
        
        return results
    
    def reverse_stress_test(
        self,
        strategies: list,
        weights: np.ndarray,
        target_loss: float = -0.10,
        n_simulations: int = 10000
    ) -> dict:
        """
        Reverse stress test: find the market conditions that cause
        the portfolio to lose exactly the target amount.
        
        Simulates random shocks and finds which combinations breach the limit.
        """
        n_strats = len(strategies)
        breaching_scenarios = []
        
        for sim in range(n_simulations):
            # Random factor shocks
            mkt_shock = self.rng.uniform(-0.50, 0.10)
            mom_shock = self.rng.uniform(-0.40, 0.20)
            val_shock = self.rng.uniform(-0.30, 0.20)
            vol_mult = self.rng.uniform(1.0, 5.0)
            
            # Quick P&L estimate
            port_pnl = 0
            for i, strat in enumerate(strategies):
                strat_pnl = (strat.market_beta * mkt_shock
                           + strat.momentum_loading * mom_shock
                           + strat.value_loading * val_shock)
                port_pnl += strat_pnl * weights[i]
            
            if port_pnl < target_loss:
                breaching_scenarios.append({
                    'market': mkt_shock,
                    'momentum': mom_shock,
                    'value': val_shock,
                    'vol_mult': vol_mult,
                    'estimated_loss': port_pnl,
                })
        
        # Find the mildest breaching scenario (most likely to occur)
        if breaching_scenarios:
            mildest = min(breaching_scenarios, 
                         key=lambda x: abs(x['market']) + abs(x['momentum']) + abs(x['value']))
        else:
            mildest = None
        
        return {
            'target_loss': target_loss,
            'n_breaching': len(breaching_scenarios),
            'breach_probability': len(breaching_scenarios) / n_simulations,
            'mildest_breach': mildest,
            'worst_breach': min(breaching_scenarios, key=lambda x: x['estimated_loss']) if breaching_scenarios else None,
        }


if __name__ == "__main__":
    from portfolio_backtester import create_pod_shop_portfolio, RiskBudgeter
    
    strategies, config = create_pod_shop_portfolio()
    weights = np.array([s.capital_allocation for s in strategies])
    weights /= np.sum(weights)
    
    print("Stress Testing & Scenario Engine")
    print("=" * 60)
    
    engine = StressTestEngine(seed=42)
    results = engine.run_all_scenarios(strategies, weights)
    
    print(f"\n{'Scenario':<22} {'Port Loss':>10} {'Max DD':>10} {'Worst Day':>10} {'Worst Pod':<22} {'Pod Loss':>10}")
    print("-" * 88)
    
    for name, res in results.items():
        print(f"{name:<22} {res['port_total_return']:>+9.1%} {res['port_max_drawdown']:>+9.1%} "
              f"{res['port_worst_day']:>+9.2%} {res['worst_pod']:<22} {res['worst_pod_return']:>+9.1%}")
    
    # Reverse stress test
    print(f"\nREVERSE STRESS TEST (target: -10% portfolio loss)")
    print("-" * 50)
    rst = engine.reverse_stress_test(strategies, weights, target_loss=-0.10)
    print(f"Breach probability: {rst['breach_probability']:.1%}")
    if rst['mildest_breach']:
        m = rst['mildest_breach']
        print(f"Mildest breach: Mkt={m['market']:+.0%}, Mom={m['momentum']:+.0%}, Val={m['value']:+.0%}")
        print(f"Estimated loss: {m['estimated_loss']:+.1%}")
