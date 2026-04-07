"""
Extension 3: Optimal Execution — TWAP / VWAP / Almgren-Chriss
================================================================
When the market-maker accumulates a large position, it needs to unwind
efficiently. This module implements:

1. TWAP (Time-Weighted Average Price): uniform execution schedule
2. VWAP (Volume-Weighted Average Price): trade proportional to volume
3. Almgren-Chriss optimal execution: minimise E[cost] + lambda*Var[cost]

The Almgren-Chriss model balances urgency risk (holding cost from
volatility) against market impact (price impact from trading).

Optimal trajectory:
    x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
    
where kappa = sqrt(lambda * sigma^2 / eta) and eta = temporary impact.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ExecutionParams:
    """Parameters for optimal execution."""
    initial_inventory: int = 5000     # Shares to unwind
    time_horizon: float = 1.0         # Hours to execute
    n_intervals: int = 60             # Trading intervals
    
    # Market parameters
    sigma: float = 0.02               # Volatility per interval
    bid_ask_spread: float = 0.01      # Half-spread in dollars
    
    # Impact parameters
    temporary_impact: float = 0.0001  # Temporary impact (eta)
    permanent_impact: float = 0.00005 # Permanent impact (gamma)
    
    # Risk aversion
    risk_aversion: float = 0.001      # Lambda in Almgren-Chriss


class TWAPExecutor:
    """Time-Weighted Average Price execution."""
    
    def schedule(self, params: ExecutionParams) -> np.ndarray:
        """Uniform execution schedule."""
        n = params.n_intervals
        trade_per_interval = params.initial_inventory / n
        return np.full(n, trade_per_interval)


class VWAPExecutor:
    """Volume-Weighted Average Price execution."""
    
    def schedule(self, params: ExecutionParams, volume_profile: np.ndarray = None) -> np.ndarray:
        """Execute proportional to expected volume."""
        n = params.n_intervals
        
        if volume_profile is None:
            # U-shaped volume profile (typical for equity markets)
            t = np.linspace(0, 1, n)
            volume_profile = 1.5 - 2.0 * t * (1 - t)  # Higher at open/close
            volume_profile = np.maximum(volume_profile, 0.3)
        
        # Normalise so total = initial_inventory
        weights = volume_profile / np.sum(volume_profile)
        return params.initial_inventory * weights


class AlmgrenChrissExecutor:
    """
    Almgren-Chriss (2001) optimal execution.
    
    Minimises: E[cost] + lambda * Var[cost]
    
    The optimal trajectory balances:
    - Urgency: sell faster to reduce volatility risk
    - Patience: sell slower to reduce market impact
    
    Solution:
        x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
        
    Trade schedule:
        n_j = x(t_j) - x(t_{j+1})
    """
    
    def schedule(self, params: ExecutionParams) -> np.ndarray:
        """Compute optimal execution trajectory."""
        X = params.initial_inventory
        T = params.n_intervals
        sigma = params.sigma
        eta = params.temporary_impact
        lam = params.risk_aversion
        
        # Kappa: trade-off parameter
        kappa = np.sqrt(lam * sigma**2 / eta)
        
        # Remaining inventory at each interval
        t = np.arange(T + 1)
        if kappa * T > 500:  # Numerical stability
            x = X * np.exp(-kappa * t)
        else:
            x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        
        # Trade sizes (difference in remaining inventory)
        trades = -np.diff(x)  # Positive = selling
        
        return trades
    
    def expected_cost(self, params: ExecutionParams) -> dict:
        """Compute expected cost and risk of the optimal strategy."""
        trades = self.schedule(params)
        
        X = params.initial_inventory
        sigma = params.sigma
        eta = params.temporary_impact
        gamma_perm = params.permanent_impact
        
        # Temporary impact cost
        temp_cost = eta * np.sum(trades**2)
        
        # Permanent impact cost
        perm_cost = 0.5 * gamma_perm * X**2
        
        # Spread cost
        spread_cost = params.bid_ask_spread * X
        
        # Volatility risk (variance of execution cost)
        T = params.n_intervals
        remaining = X - np.cumsum(np.insert(trades, 0, 0))[:-1]
        vol_risk = sigma**2 * np.sum(remaining**2)
        
        total_cost = temp_cost + perm_cost + spread_cost
        
        return {
            'total_cost': total_cost,
            'temporary_impact_cost': temp_cost,
            'permanent_impact_cost': perm_cost,
            'spread_cost': spread_cost,
            'volatility_risk': vol_risk,
            'cost_bps': total_cost / (X * 100) * 10000,  # Assuming $100 price
            'trades': trades,
        }


class ExecutionSimulator:
    """Simulate execution with realistic price dynamics."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def simulate(
        self,
        trades: np.ndarray,
        params: ExecutionParams,
        n_sims: int = 10000
    ) -> dict:
        """
        Monte Carlo simulation of execution outcomes.
        
        Returns distribution of execution costs (implementation shortfall).
        """
        n = len(trades)
        costs = np.zeros(n_sims)
        
        arrival_price = 100.0  # Reference price
        
        for sim in range(n_sims):
            price = arrival_price
            total_proceeds = 0.0
            
            for i in range(n):
                # Price evolution
                price += params.sigma * self.rng.standard_normal()
                
                # Permanent impact
                price -= params.permanent_impact * trades[i]
                
                # Execution price (with temporary impact)
                exec_price = price - params.temporary_impact * trades[i] - params.bid_ask_spread
                
                total_proceeds += exec_price * trades[i]
            
            # Implementation shortfall
            ideal_proceeds = arrival_price * params.initial_inventory
            costs[sim] = ideal_proceeds - total_proceeds
        
        return {
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'cost_bps': np.mean(costs) / (params.initial_inventory * arrival_price) * 10000,
            'var_95': np.percentile(costs, 95),
            'cost_distribution': costs,
        }


def compare_strategies(params: ExecutionParams = None) -> dict:
    """Compare TWAP, VWAP, and Almgren-Chriss execution."""
    if params is None:
        params = ExecutionParams()
    
    twap = TWAPExecutor().schedule(params)
    vwap = VWAPExecutor().schedule(params)
    ac = AlmgrenChrissExecutor().schedule(params)
    
    sim = ExecutionSimulator(seed=42)
    
    results = {}
    for name, trades in [('TWAP', twap), ('VWAP', vwap), ('Almgren-Chriss', ac)]:
        sim_result = sim.simulate(trades, params)
        results[name] = {
            'trades': trades,
            'mean_cost_bps': sim_result['cost_bps'],
            'std_cost': sim_result['std_cost'],
            'var_95': sim_result['var_95'],
        }
    
    return results


if __name__ == "__main__":
    print("Optimal Execution Extension")
    print("=" * 50)
    
    params = ExecutionParams(
        initial_inventory=5000,
        n_intervals=60,
        sigma=0.02,
        temporary_impact=0.0001,
        permanent_impact=0.00005,
        risk_aversion=0.001
    )
    
    # Almgren-Chriss analytics
    ac = AlmgrenChrissExecutor()
    cost_analysis = ac.expected_cost(params)
    
    print(f"\nALMGREN-CHRISS ANALYTICS")
    print(f"Total cost:       ${cost_analysis['total_cost']:,.2f}")
    print(f"  Temporary:      ${cost_analysis['temporary_impact_cost']:,.2f}")
    print(f"  Permanent:      ${cost_analysis['permanent_impact_cost']:,.2f}")
    print(f"  Spread:         ${cost_analysis['spread_cost']:,.2f}")
    print(f"Cost (bps):       {cost_analysis['cost_bps']:.1f}")
    
    # Compare strategies
    print(f"\nSTRATEGY COMPARISON (Monte Carlo)")
    print(f"{'Strategy':<18} {'Cost (bps)':>10} {'Std ($)':>12} {'VaR 95 ($)':>12}")
    print("-" * 54)
    
    results = compare_strategies(params)
    for name, res in results.items():
        print(f"{name:<18} {res['mean_cost_bps']:>9.1f} {res['std_cost']:>11,.0f} {res['var_95']:>11,.0f}")
    
    # Risk aversion sensitivity
    print(f"\nRISK AVERSION SENSITIVITY")
    print(f"{'Lambda':<10} {'Front-loading':>15} {'Cost (bps)':>12}")
    print("-" * 40)
    
    for lam in [0.0001, 0.001, 0.01, 0.1]:
        p = ExecutionParams(risk_aversion=lam)
        trades = ac.schedule(p)
        front_pct = np.sum(trades[:15]) / np.sum(trades)
        cost = ac.expected_cost(p)
        print(f"{lam:<10.4f} {front_pct:>14.1%} {cost['cost_bps']:>11.1f}")
