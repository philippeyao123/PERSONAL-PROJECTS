"""
Extension 1: Multi-Asset Market-Making
========================================
Market-making on 2+ correlated assets with cross-asset hedging.

When assets are correlated, inventory in asset A creates implicit exposure
to asset B. The multi-asset MM must:
1. Quote each asset with an AS-style quoter
2. Compute portfolio-level inventory risk
3. Hedge cross-asset exposure by skewing quotes

The reservation price becomes vector-valued:
    r_i = s_i - gamma * sum_j (Sigma_{ij} * q_j) * (T - t)

where Sigma is the covariance matrix of mid-price returns.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import time


@dataclass
class MultiAssetParams:
    """Parameters for multi-asset market-making."""
    n_assets: int = 2
    initial_prices: np.ndarray = None
    sigmas: np.ndarray = None            # Per-asset volatility
    correlation: np.ndarray = None       # Correlation matrix
    tick_sizes: np.ndarray = None
    lambda_rates: np.ndarray = None      # Order arrival per asset
    
    def __post_init__(self):
        n = self.n_assets
        if self.initial_prices is None:
            self.initial_prices = np.array([100.0, 50.0][:n])
        if self.sigmas is None:
            self.sigmas = np.array([0.02, 0.025][:n])
        if self.correlation is None:
            self.correlation = np.array([[1.0, 0.7], [0.7, 1.0]])[:n, :n]
        if self.tick_sizes is None:
            self.tick_sizes = np.full(n, 0.01)
        if self.lambda_rates is None:
            self.lambda_rates = np.full(n, 5.0)
    
    @property
    def covariance(self) -> np.ndarray:
        """Covariance matrix from sigmas and correlation."""
        D = np.diag(self.sigmas)
        return D @ self.correlation @ D
    
    @property
    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.covariance)


@dataclass
class MultiAssetState:
    """State for multi-asset simulation."""
    prices: np.ndarray = None
    inventories: np.ndarray = None
    cash: float = 0.0
    step: int = 0
    
    # History
    price_history: list = field(default_factory=list)
    inventory_history: list = field(default_factory=list)
    pnl_history: list = field(default_factory=list)
    spread_history: list = field(default_factory=list)
    hedge_ratio_history: list = field(default_factory=list)
    
    @property
    def mark_to_market(self) -> float:
        if self.prices is not None and self.inventories is not None:
            return self.cash + np.dot(self.inventories, self.prices)
        return self.cash
    
    @property
    def portfolio_risk(self) -> float:
        """Portfolio variance = q' * Sigma * q."""
        return 0.0  # Computed externally with covariance


class MultiAssetQuoter:
    """
    Multi-asset Avellaneda-Stoikov quoter with cross-hedging.
    
    The key insight: in multi-asset MM, inventory in one asset affects
    the optimal quote in another via the covariance matrix.
    
    Reservation price vector:
        r_i = s_i - gamma * sum_j(Cov_{ij} * q_j) * tau
    
    This naturally skews quotes to reduce portfolio-level risk.
    """
    
    def __init__(self, gamma: float = 0.01, k: float = 1.5, T: float = 1/252):
        self.gamma = gamma
        self.k = k
        self.T = T
    
    def compute_quotes(
        self,
        prices: np.ndarray,
        inventories: np.ndarray,
        covariance: np.ndarray,
        time_remaining: float,
        max_spread_pct: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bid/ask for all assets simultaneously.
        
        Returns:
            bids: (n_assets,) bid prices
            asks: (n_assets,) ask prices
        """
        n = len(prices)
        tau = max(time_remaining, 1e-6)
        
        # Portfolio-adjusted reservation prices
        # r_i = s_i - gamma * (Cov @ q)_i * tau
        cov_q = covariance @ inventories
        reservations = prices - self.gamma * cov_q * tau
        
        # Per-asset spread (using diagonal variance)
        bids = np.zeros(n)
        asks = np.zeros(n)
        
        for i in range(n):
            var_i = covariance[i, i]
            spread = self.gamma * var_i * tau + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
            half_spread = np.clip(spread / 2, 0.0005 * prices[i], max_spread_pct * prices[i])
            
            bids[i] = np.floor((reservations[i] - half_spread) / 0.01) * 0.01
            asks[i] = np.ceil((reservations[i] + half_spread) / 0.01) * 0.01
        
        return bids, asks
    
    def compute_hedge_ratios(
        self,
        inventories: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Compute optimal hedge ratios (beta) between assets.
        
        Beta_ij = Cov(i,j) / Var(j): units of asset j to hedge 1 unit of asset i.
        """
        n = len(inventories)
        betas = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    betas[i, j] = covariance[i, j] / covariance[j, j]
        return betas


class MultiAssetSimulator:
    """Full multi-asset market-making simulation."""
    
    def __init__(self, params: MultiAssetParams = None, gamma: float = 0.01, seed: int = 42):
        self.params = params or MultiAssetParams()
        self.quoter = MultiAssetQuoter(gamma=gamma)
        self.rng = np.random.default_rng(seed)
        self.L = self.params.cholesky
    
    def run(self, n_steps: int = 2340) -> MultiAssetState:
        """Run multi-asset simulation."""
        p = self.params
        n = p.n_assets
        
        state = MultiAssetState(
            prices=p.initial_prices.copy(),
            inventories=np.zeros(n, dtype=int)
        )
        
        for step in range(n_steps):
            state.step = step
            tau = max((n_steps - step) / n_steps * self.quoter.T, 1e-8)
            
            # 1. Evolve correlated prices
            eps = self.rng.standard_normal(n)
            dW = self.L @ eps
            state.prices += dW
            state.prices = np.maximum(state.prices, 0.01)
            
            # 2. Compute quotes
            bids, asks = self.quoter.compute_quotes(
                state.prices, state.inventories, p.covariance, tau
            )
            
            # 3. Generate order flow per asset
            for i in range(n):
                n_buy = self.rng.poisson(p.lambda_rates[i])
                n_sell = self.rng.poisson(p.lambda_rates[i])
                
                for _ in range(n_buy):
                    aggressor_price = state.prices[i] + self.rng.exponential(0.005) * state.prices[i]
                    if aggressor_price >= asks[i]:
                        size = max(10, int(self.rng.normal(100, 30)))
                        state.cash += asks[i] * size
                        state.inventories[i] -= size
                
                for _ in range(n_sell):
                    aggressor_price = state.prices[i] - self.rng.exponential(0.005) * state.prices[i]
                    if aggressor_price <= bids[i]:
                        size = max(10, int(self.rng.normal(100, 30)))
                        state.cash -= bids[i] * size
                        state.inventories[i] += size
            
            # 4. Record
            state.price_history.append(state.prices.copy())
            state.inventory_history.append(state.inventories.copy())
            state.pnl_history.append(state.mark_to_market)
            state.spread_history.append(asks - bids)
            
            betas = self.quoter.compute_hedge_ratios(state.inventories, p.covariance)
            state.hedge_ratio_history.append(betas.copy())
        
        return state


if __name__ == "__main__":
    params = MultiAssetParams(
        n_assets=2,
        initial_prices=np.array([100.0, 50.0]),
        sigmas=np.array([0.02, 0.025]),
        correlation=np.array([[1.0, 0.7], [0.7, 1.0]])
    )
    
    print("Multi-Asset Market-Making Extension")
    print("=" * 50)
    print(f"Assets: {params.n_assets}")
    print(f"Correlation:\n{params.correlation}")
    print(f"Covariance:\n{params.covariance}")
    
    t0 = time.time()
    sim = MultiAssetSimulator(params, gamma=0.01, seed=42)
    state = sim.run(n_steps=2340)
    print(f"\nSimulation: {time.time()-t0:.2f}s")
    
    print(f"\nRESULTS")
    print(f"Total P&L:       ${state.mark_to_market:,.2f}")
    print(f"Final inventory: {state.inventories}")
    print(f"Final prices:    {state.prices}")
    
    # Portfolio risk
    q = state.inventories.astype(float)
    port_var = q @ params.covariance @ q
    print(f"Portfolio var:   {port_var:,.2f}")
    
    # Hedge ratios
    betas = sim.quoter.compute_hedge_ratios(state.inventories, params.covariance)
    print(f"Hedge ratio (asset1 vs asset2): {betas[0,1]:.3f}")
    print(f"Hedge ratio (asset2 vs asset1): {betas[1,0]:.3f}")
