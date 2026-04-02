"""
Extension 4: Liquidity Risk
==============================
Models the constraints and costs of trading illiquid positions:

  1. Position sizing: constrained by ADV (average daily volume)
  2. Liquidation cost under stress: how much does forced selling cost?
  3. Crowding risk: when everyone holds the same positions, exits are expensive
  4. Liquidity-adjusted VaR: standard VaR underestimates risk for illiquid books

Key insight: liquidity is not constant. It evaporates precisely when
you need it most (crises). Pod shops care deeply about this because
forced deleveraging destroys alpha.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import time


@dataclass
class LiquidityProfile:
    """Liquidity characteristics per strategy."""
    name: str
    avg_position_pct_adv: float     # Average position as % of ADV
    max_position_pct_adv: float     # Largest position as % of ADV
    n_positions: int                 # Number of positions
    avg_spread_bps: float           # Average bid-ask spread
    concentration_top5: float       # Top 5 positions as % of book
    
    @property
    def days_to_liquidate(self) -> float:
        """Estimated days to fully liquidate at 10% ADV participation."""
        return self.max_position_pct_adv / 10.0
    
    @staticmethod
    def from_style(name: str, style: str, capital: float) -> 'LiquidityProfile':
        profiles = {
            'stat_arb': LiquidityProfile(name, 2.0, 5.0, 200, 3.0, 0.15),
            'equity_ls': LiquidityProfile(name, 5.0, 15.0, 50, 5.0, 0.35),
            'macro': LiquidityProfile(name, 1.0, 3.0, 20, 1.0, 0.50),
            'mean_reversion': LiquidityProfile(name, 3.0, 8.0, 80, 4.0, 0.20),
            'momentum': LiquidityProfile(name, 4.0, 12.0, 40, 2.0, 0.30),
        }
        return profiles.get(style, LiquidityProfile(name, 3.0, 10.0, 50, 3.0, 0.25))


class LiquidationCostModel:
    """
    Estimate the cost of liquidating positions under different conditions.
    
    Normal liquidation: orderly, over multiple days
    Stress liquidation: forced, concurrent with other sellers
    Fire sale: immediate, maximum impact
    """
    
    @staticmethod
    def normal_liquidation(
        position_pct_adv: float,
        spread_bps: float,
        participation_rate: float = 0.10
    ) -> dict:
        """Cost of orderly liquidation."""
        days = position_pct_adv / (participation_rate * 100)
        
        # Impact: sqrt model (Almgren)
        daily_impact_bps = 5.0 * np.sqrt(participation_rate / 0.10)
        total_impact_bps = daily_impact_bps * days
        
        total_cost_bps = spread_bps + total_impact_bps
        
        return {
            'days': days,
            'spread_cost_bps': spread_bps,
            'impact_cost_bps': total_impact_bps,
            'total_cost_bps': total_cost_bps,
            'scenario': 'normal',
        }
    
    @staticmethod
    def stress_liquidation(
        position_pct_adv: float,
        spread_bps: float,
        participation_rate: float = 0.10,
        stress_multiplier: float = 3.0
    ) -> dict:
        """Cost when liquidating under market stress."""
        # Spreads widen
        stress_spread = spread_bps * stress_multiplier
        
        # Volume drops (ADV shrinks by 40%)
        effective_adv_mult = 0.6
        effective_pct_adv = position_pct_adv / effective_adv_mult
        
        days = effective_pct_adv / (participation_rate * 100)
        
        # Impact amplified
        daily_impact_bps = 5.0 * np.sqrt(participation_rate / 0.10) * stress_multiplier
        total_impact_bps = daily_impact_bps * days
        
        return {
            'days': days,
            'spread_cost_bps': stress_spread,
            'impact_cost_bps': total_impact_bps,
            'total_cost_bps': stress_spread + total_impact_bps,
            'scenario': 'stress',
        }
    
    @staticmethod
    def fire_sale(
        position_pct_adv: float,
        spread_bps: float
    ) -> dict:
        """Immediate liquidation at any price."""
        # Full impact: linear in position size
        impact_bps = 20.0 * position_pct_adv / 100  # 20bps per 1% ADV
        total_cost_bps = spread_bps * 5 + impact_bps
        
        return {
            'days': 1,
            'spread_cost_bps': spread_bps * 5,
            'impact_cost_bps': impact_bps,
            'total_cost_bps': total_cost_bps,
            'scenario': 'fire_sale',
        }


class CrowdingRiskModel:
    """
    Estimate crowding risk: correlated positioning across managers.
    
    When many managers hold the same positions (factor crowding),
    simultaneous deleveraging creates a positive feedback loop:
    selling -> price drop -> more selling -> more price drop.
    
    Crowding score = correlation of strategy returns with factor returns
    * factor popularity.
    """
    
    @staticmethod
    def compute_crowding_score(
        strategy_returns: np.ndarray,
        factor_returns: np.ndarray,
        factor_popularity: float = 0.5
    ) -> float:
        """
        Crowding score [0, 1].
        
        High score = strategy is crowded, vulnerable to unwinds.
        """
        if len(strategy_returns) < 20:
            return 0.0
        
        corr = np.corrcoef(strategy_returns, factor_returns)[0, 1]
        crowding = abs(corr) * factor_popularity
        return min(crowding, 1.0)
    
    @staticmethod
    def unwind_impact(
        crowding_score: float,
        base_liquidation_cost: float
    ) -> float:
        """
        Additional cost from crowded unwind.
        
        When crowding is high, liquidation cost multiplies because
        everyone is selling the same thing.
        """
        multiplier = 1.0 + 3.0 * crowding_score**2  # Quadratic in crowding
        return base_liquidation_cost * multiplier


class LiquidityAdjustedRisk:
    """
    Liquidity-adjusted Value at Risk (LVaR).
    
    LVaR = VaR + Liquidation cost under stress
    
    Standard VaR assumes positions can be unwound at mid-price.
    LVaR accounts for the cost of actually exiting.
    """
    
    @staticmethod
    def compute_lvar(
        returns: np.ndarray,
        liquidation_cost_pct: float,
        confidence: float = 0.99
    ) -> dict:
        """Compute LVaR."""
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = np.mean(returns[returns <= var])
        
        lvar = var - liquidation_cost_pct
        lcvar = cvar - liquidation_cost_pct
        
        return {
            'var': var,
            'cvar': cvar,
            'lvar': lvar,
            'lcvar': lcvar,
            'liquidity_addon': liquidation_cost_pct,
            'lvar_ratio': lvar / var if var < 0 else 1.0,
        }


class LiquidityRiskAnalyser:
    """Full liquidity risk analysis for the portfolio."""
    
    def __init__(self):
        self.liquidation = LiquidationCostModel()
        self.crowding = CrowdingRiskModel()
        self.lvar = LiquidityAdjustedRisk()
    
    def analyse(
        self,
        strategies: list,
        returns: np.ndarray,
        factors: dict,
        weights: np.ndarray,
        total_capital: float = 100_000_000
    ) -> dict:
        """Full liquidity risk analysis."""
        n_strats = len(strategies)
        results = {}
        
        for i, strat in enumerate(strategies):
            capital = total_capital * weights[i]
            liq = LiquidityProfile.from_style(strat.name, strat.style, capital)
            
            # Liquidation costs under different scenarios
            normal = self.liquidation.normal_liquidation(liq.avg_position_pct_adv, liq.avg_spread_bps)
            stress = self.liquidation.stress_liquidation(liq.avg_position_pct_adv, liq.avg_spread_bps)
            fire = self.liquidation.fire_sale(liq.avg_position_pct_adv, liq.avg_spread_bps)
            
            # Crowding
            market_factor = factors.get('market', np.zeros(len(returns)))
            min_len = min(len(returns[:, i]), len(market_factor))
            crowd_score = self.crowding.compute_crowding_score(
                returns[:min_len, i], market_factor[:min_len], 0.5
            )
            
            # LVaR
            stress_cost_pct = stress['total_cost_bps'] / 10000
            lvar_result = self.lvar.compute_lvar(returns[:, i], stress_cost_pct)
            
            results[strat.name] = {
                'liquidity_profile': liq,
                'normal_cost': normal,
                'stress_cost': stress,
                'fire_sale_cost': fire,
                'crowding_score': crowd_score,
                'lvar': lvar_result,
                'days_to_liquidate_normal': normal['days'],
                'days_to_liquidate_stress': stress['days'],
            }
        
        # Portfolio-level LVaR
        port_returns = returns @ weights
        avg_stress_cost = np.mean([r['stress_cost']['total_cost_bps'] for r in results.values()]) / 10000
        port_lvar = self.lvar.compute_lvar(port_returns, avg_stress_cost)
        
        return {
            'strategy_results': results,
            'portfolio_lvar': port_lvar,
            'portfolio_avg_stress_cost_bps': avg_stress_cost * 10000,
        }


if __name__ == "__main__":
    from portfolio_backtester import create_pod_shop_portfolio, FactorModel
    
    strategies, config = create_pod_shop_portfolio()
    weights = np.array([s.capital_allocation for s in strategies])
    weights /= np.sum(weights)
    
    fm = FactorModel(seed=42)
    returns, factors = fm.simulate_strategy_returns(strategies, 756)
    
    print("Liquidity Risk Analysis")
    print("=" * 60)
    
    analyser = LiquidityRiskAnalyser()
    results = analyser.analyse(strategies, returns, factors, weights)
    
    print(f"\nLIQUIDATION COST BY SCENARIO")
    print(f"{'Strategy':<25} {'Normal':>10} {'Stress':>10} {'Fire Sale':>10} {'Crowd':>8}")
    print("-" * 65)
    for name, res in results['strategy_results'].items():
        print(f"{name:<25} {res['normal_cost']['total_cost_bps']:>8.0f}bp "
              f"{res['stress_cost']['total_cost_bps']:>8.0f}bp "
              f"{res['fire_sale_cost']['total_cost_bps']:>8.0f}bp "
              f"{res['crowding_score']:>7.2f}")
    
    print(f"\nDAYS TO LIQUIDATE")
    print(f"{'Strategy':<25} {'Normal':>10} {'Stress':>10}")
    print("-" * 47)
    for name, res in results['strategy_results'].items():
        print(f"{name:<25} {res['days_to_liquidate_normal']:>8.1f}d {res['days_to_liquidate_stress']:>8.1f}d")
    
    print(f"\nLIQUIDITY-ADJUSTED VAR (99%)")
    print(f"{'Strategy':<25} {'VaR':>10} {'LVaR':>10} {'Addon':>10} {'Ratio':>8}")
    print("-" * 65)
    for name, res in results['strategy_results'].items():
        lv = res['lvar']
        print(f"{name:<25} {lv['var']:>9.2%} {lv['lvar']:>9.2%} {lv['liquidity_addon']:>9.2%} {lv['lvar_ratio']:>7.2f}x")
    
    plv = results['portfolio_lvar']
    print(f"\n{'PORTFOLIO':<25} {plv['var']:>9.2%} {plv['lvar']:>9.2%} {plv['liquidity_addon']:>9.2%} {plv['lvar_ratio']:>7.2f}x")
    print(f"\nAvg stress liquidation cost: {results['portfolio_avg_stress_cost_bps']:.0f} bps")
