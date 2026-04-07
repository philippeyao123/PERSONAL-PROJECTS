"""
Extension 3: Transaction Costs & Slippage Model
=================================================
Models the real-world friction that erodes strategy alpha:

  1. Commission: fixed per-share or per-notional fee
  2. Spread cost: crossing the bid-ask spread
  3. Market impact: price moves against you as you trade (temporary + permanent)
  4. Financing cost: cost of leverage (short rebate, margin interest)
  5. Slippage: difference between signal price and execution price

The net Sharpe after costs is what actually matters.
Many strategies with gross Sharpe > 1 have net Sharpe < 0.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time


@dataclass
class CostModel:
    """Transaction cost parameters."""
    commission_bps: float = 1.0         # Commission in bps of notional traded
    spread_bps: float = 3.0             # Half-spread cost per trade
    impact_bps_per_pct_adv: float = 5.0 # Market impact per 1% of ADV traded
    financing_rate_long: float = 0.05   # Annual cost of leverage (long)
    financing_rate_short: float = 0.06  # Annual cost of short financing
    slippage_bps: float = 1.0           # Random execution slippage
    
    @property
    def total_one_way_bps(self) -> float:
        """Approximate total one-way cost."""
        return self.commission_bps + self.spread_bps + self.slippage_bps


@dataclass
class TurnoverProfile:
    """Turnover characteristics per strategy style."""
    style: str
    daily_turnover_pct: float    # % of portfolio traded per day
    avg_trade_size_pct_adv: float  # Average trade as % of ADV
    
    @staticmethod
    def from_style(style: str) -> 'TurnoverProfile':
        profiles = {
            'stat_arb':       TurnoverProfile('stat_arb', 0.15, 0.5),
            'equity_ls':      TurnoverProfile('equity_ls', 0.03, 1.0),
            'macro':          TurnoverProfile('macro', 0.02, 2.0),
            'mean_reversion': TurnoverProfile('mean_reversion', 0.10, 0.8),
            'momentum':       TurnoverProfile('momentum', 0.05, 1.5),
        }
        return profiles.get(style, TurnoverProfile(style, 0.05, 1.0))


class TransactionCostEngine:
    """
    Compute transaction costs and their impact on portfolio performance.
    
    For each strategy, costs are:
      Total cost = turnover * (commission + spread + impact + slippage) + financing
    """
    
    def __init__(self, cost_model: CostModel = None):
        self.cost = cost_model or CostModel()
    
    def compute_strategy_costs(
        self,
        strategy_name: str,
        style: str,
        gross_returns: np.ndarray,
        capital: float,
        leverage: float = 1.0
    ) -> dict:
        """Compute daily costs for a single strategy."""
        n_days = len(gross_returns)
        turnover = TurnoverProfile.from_style(style)
        
        daily_notional_traded = capital * leverage * turnover.daily_turnover_pct
        
        # Per-trade costs (each direction)
        commission_daily = daily_notional_traded * self.cost.commission_bps / 10000
        spread_daily = daily_notional_traded * self.cost.spread_bps / 10000
        
        # Market impact (increases with trade size relative to ADV)
        impact_daily = daily_notional_traded * self.cost.impact_bps_per_pct_adv * turnover.avg_trade_size_pct_adv / 10000
        
        # Slippage (random component)
        rng = np.random.default_rng(42)
        slippage_daily = daily_notional_traded * self.cost.slippage_bps / 10000 * (1 + rng.normal(0, 0.3, n_days))
        slippage_daily = np.maximum(slippage_daily, 0)
        
        # Financing (daily)
        long_financing = capital * leverage * 0.5 * self.cost.financing_rate_long / 252  # Assume 50% long
        short_financing = capital * leverage * 0.5 * self.cost.financing_rate_short / 252
        financing_daily = long_financing + short_financing
        
        # Total daily cost
        total_daily = commission_daily + spread_daily + impact_daily + slippage_daily + financing_daily
        
        # Cost as fraction of capital (for return adjustment)
        cost_return = total_daily / capital
        
        # Net returns
        net_returns = gross_returns - cost_return
        
        # Annual cost
        annual_cost_pct = np.mean(cost_return) * 252
        
        return {
            'name': strategy_name,
            'gross_returns': gross_returns,
            'net_returns': net_returns,
            'daily_cost': total_daily,
            'cost_return': cost_return,
            'annual_cost_pct': annual_cost_pct,
            'breakdown': {
                'commission': commission_daily * 252 / capital,
                'spread': spread_daily * 252 / capital,
                'impact': impact_daily * 252 / capital,
                'slippage': np.mean(slippage_daily) * 252 / capital,
                'financing': financing_daily * 252 / capital,
            },
            'turnover_daily': turnover.daily_turnover_pct,
            'gross_sharpe': np.mean(gross_returns) / max(np.std(gross_returns), 1e-10) * np.sqrt(252),
            'net_sharpe': np.mean(net_returns) / max(np.std(net_returns), 1e-10) * np.sqrt(252),
        }
    
    def analyse_portfolio(
        self,
        strategies: list,
        returns: np.ndarray,
        weights: np.ndarray,
        total_capital: float = 100_000_000
    ) -> dict:
        """Full portfolio cost analysis."""
        n_strats = len(strategies)
        strat_results = []
        
        for i, strat in enumerate(strategies):
            capital = total_capital * weights[i]
            result = self.compute_strategy_costs(
                strat.name, strat.style,
                returns[:, i],
                capital,
                leverage=strat.max_gross_leverage * 0.5  # Assume 50% of max
            )
            strat_results.append(result)
        
        # Portfolio-level
        gross_port = returns @ weights
        net_port = np.zeros_like(gross_port)
        total_cost = np.zeros_like(gross_port)
        
        for i, res in enumerate(strat_results):
            net_port += res['net_returns'] * weights[i]
            total_cost += res['cost_return'] * weights[i]
        
        gross_sharpe = np.mean(gross_port) / max(np.std(gross_port), 1e-10) * np.sqrt(252)
        net_sharpe = np.mean(net_port) / max(np.std(net_port), 1e-10) * np.sqrt(252)
        
        return {
            'strategy_costs': strat_results,
            'gross_port_returns': gross_port,
            'net_port_returns': net_port,
            'total_cost_return': total_cost,
            'gross_sharpe': gross_sharpe,
            'net_sharpe': net_sharpe,
            'sharpe_decay': gross_sharpe - net_sharpe,
            'annual_cost_pct': np.mean(total_cost) * 252,
            'total_annual_cost': np.mean(total_cost) * 252 * total_capital,
        }
    
    def breakeven_analysis(
        self,
        strategies: list,
        returns: np.ndarray,
        weights: np.ndarray,
        total_capital: float = 100_000_000
    ) -> dict:
        """Find the cost level at which each strategy becomes unprofitable."""
        results = {}
        
        for i, strat in enumerate(strategies):
            gross_return = np.mean(returns[:, i]) * 252
            capital = total_capital * weights[i]
            turnover = TurnoverProfile.from_style(strat.style)
            
            # Total cost = turnover * cost_per_trade * leverage
            leverage = strat.max_gross_leverage * 0.5
            daily_traded = capital * leverage * turnover.daily_turnover_pct
            annual_traded = daily_traded * 252
            
            # Breakeven: gross_return = total_cost
            # total_cost = annual_traded * breakeven_bps / 10000 + financing
            financing = capital * leverage * 0.5 * (self.cost.financing_rate_long + self.cost.financing_rate_short) / 2
            net_for_trading = gross_return * capital - financing
            
            if annual_traded > 0:
                breakeven_bps = net_for_trading / annual_traded * 10000
            else:
                breakeven_bps = float('inf')
            
            results[strat.name] = {
                'gross_return': gross_return,
                'breakeven_bps': breakeven_bps,
                'current_cost_bps': self.cost.total_one_way_bps,
                'margin_bps': breakeven_bps - self.cost.total_one_way_bps,
                'at_risk': breakeven_bps < self.cost.total_one_way_bps * 1.5,
            }
        
        return results


if __name__ == "__main__":
    from portfolio_backtester import create_pod_shop_portfolio, FactorModel
    
    strategies, config = create_pod_shop_portfolio()
    weights = np.array([s.capital_allocation for s in strategies])
    weights /= np.sum(weights)
    
    fm = FactorModel(seed=42)
    returns, _ = fm.simulate_strategy_returns(strategies, 756)
    
    print("Transaction Costs & Slippage Model")
    print("=" * 55)
    
    engine = TransactionCostEngine()
    results = engine.analyse_portfolio(strategies, returns, weights)
    
    print(f"\nSTRATEGY-LEVEL COSTS")
    print(f"{'Strategy':<25} {'Gross SR':>9} {'Net SR':>9} {'Cost pa':>9} {'Turnover':>9}")
    print("-" * 63)
    for res in results['strategy_costs']:
        print(f"{res['name']:<25} {res['gross_sharpe']:>8.2f} {res['net_sharpe']:>8.2f} "
              f"{res['annual_cost_pct']:>8.1%} {res['turnover_daily']:>8.1%}/d")
    
    print(f"\nPORTFOLIO")
    print(f"Gross Sharpe:  {results['gross_sharpe']:.2f}")
    print(f"Net Sharpe:    {results['net_sharpe']:.2f}")
    print(f"Sharpe decay:  {results['sharpe_decay']:.2f}")
    print(f"Annual cost:   {results['annual_cost_pct']:.2%} (${results['total_annual_cost']:,.0f})")
    
    # Cost breakdown
    print(f"\nCOST BREAKDOWN (first strategy)")
    bd = results['strategy_costs'][0]['breakdown']
    for comp, val in bd.items():
        print(f"  {comp:<15} {val:.2%} pa")
    
    # Breakeven
    print(f"\nBREAKEVEN ANALYSIS")
    print(f"{'Strategy':<25} {'Gross Ret':>10} {'BE (bps)':>10} {'Current':>10} {'Margin':>10} {'Risk':>6}")
    print("-" * 73)
    be = engine.breakeven_analysis(strategies, returns, weights)
    for name, b in be.items():
        risk = "YES" if b['at_risk'] else "no"
        print(f"{name:<25} {b['gross_return']:>9.1%} {b['breakeven_bps']:>9.1f} "
              f"{b['current_cost_bps']:>9.1f} {b['margin_bps']:>9.1f} {risk:>6}")
