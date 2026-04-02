"""
Market-Making Simulator — Avellaneda-Stoikov Framework
=======================================================
Simulates an electronic market-maker quoting bid/ask prices on a single
asset, managing inventory risk under stochastic mid-price dynamics.

Core model (Avellaneda & Stoikov, 2008):
    Reservation price:  r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    Optimal spread:     delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

where:
    s     = mid-price
    q     = inventory (signed)
    gamma = risk aversion parameter
    sigma = mid-price volatility
    T     = terminal time
    k     = order arrival intensity parameter

The market-maker faces:
- Adverse selection: informed flow moves the mid against inventory
- Inventory risk: holding large positions exposes to price moves
- Execution uncertainty: limit orders may not fill

This simulator models all three and analyses P&L across vol regimes.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


# ============================================================================
# Market Microstructure Parameters
# ============================================================================

@dataclass
class MarketParams:
    """Market microstructure parameters."""
    initial_price: float = 100.0
    sigma: float = 0.02              # Mid-price volatility (per step)
    tick_size: float = 0.01          # Minimum price increment
    dt: float = 1.0 / (252 * 390)   # 1 minute in trading years (252 days, 390 min/day)
    
    # Order flow
    lambda_buy: float = 5.0          # Buy order arrival rate (per step)
    lambda_sell: float = 5.0         # Sell order arrival rate (per step)
    order_size_mean: float = 100     # Mean order size
    order_size_std: float = 30       # Std of order size
    
    # Adverse selection
    informed_fraction: float = 0.15  # Fraction of informed traders
    informed_edge: float = 0.005     # Information advantage (price impact)
    
    # Market impact
    temporary_impact: float = 0.0001 # Temporary price impact per unit
    permanent_impact: float = 0.00005 # Permanent price impact per unit


@dataclass
class MMStrategy:
    """Market-making strategy parameters (Avellaneda-Stoikov)."""
    gamma: float = 0.01              # Risk aversion
    k: float = 1.5                   # Order arrival intensity parameter
    max_inventory: int = 500         # Maximum absolute inventory
    max_spread: float = 0.05         # Maximum half-spread (% of mid)
    min_spread: float = 0.0005       # Minimum half-spread
    order_size: int = 100            # Quote size
    
    # Inventory skew
    skew_factor: float = 0.001       # Price adjustment per unit of inventory
    
    # Terminal time for AS model (in trading years)
    T: float = 1.0 / 252             # 1 trading day


# ============================================================================
# Order Book
# ============================================================================

@dataclass
class Order:
    """Limit order."""
    side: str               # 'bid' or 'ask'
    price: float
    size: int
    timestamp: int


@dataclass
class Fill:
    """Execution record."""
    timestamp: int
    side: str               # 'buy' or 'sell' (from MM perspective)
    price: float
    size: int
    mid_at_fill: float
    inventory_after: int


class OrderBook:
    """Simplified order book for the market-maker's quotes."""
    
    def __init__(self):
        self.bid: Optional[Order] = None
        self.ask: Optional[Order] = None
        self.fills: List[Fill] = []
    
    def place_quotes(self, bid_price: float, ask_price: float, 
                     size: int, timestamp: int):
        """Place/replace bid and ask quotes."""
        self.bid = Order('bid', bid_price, size, timestamp)
        self.ask = Order('ask', ask_price, size, timestamp)
    
    def check_fills(self, incoming_price: float, incoming_side: str,
                    incoming_size: int, mid_price: float, 
                    timestamp: int, inventory: int) -> List[Fill]:
        """
        Check if incoming order fills against our quotes.
        
        incoming_side: 'buy' or 'sell' from the aggressor's perspective
        """
        fills = []
        
        if incoming_side == 'buy' and self.ask is not None:
            # Aggressor buys, we sell at our ask
            if incoming_price >= self.ask.price:
                fill_size = min(incoming_size, self.ask.size)
                fill = Fill(
                    timestamp=timestamp,
                    side='sell',  # We sell
                    price=self.ask.price,
                    size=fill_size,
                    mid_at_fill=mid_price,
                    inventory_after=inventory - fill_size
                )
                fills.append(fill)
                self.fills.append(fill)
        
        elif incoming_side == 'sell' and self.bid is not None:
            # Aggressor sells, we buy at our bid
            if incoming_price <= self.bid.price:
                fill_size = min(incoming_size, self.bid.size)
                fill = Fill(
                    timestamp=timestamp,
                    side='buy',  # We buy
                    price=self.bid.price,
                    size=fill_size,
                    mid_at_fill=mid_price,
                    inventory_after=inventory + fill_size
                )
                fills.append(fill)
                self.fills.append(fill)
        
        return fills


# ============================================================================
# Mid-Price Process
# ============================================================================

class MidPriceProcess:
    """
    Arithmetic Brownian motion mid-price with regime switching.
    
    Two regimes:
    - Low vol:  sigma_low, with mean-reverting drift
    - High vol: sigma_high, with momentum
    
    Regime transitions via Markov chain.
    """
    
    def __init__(self, params: MarketParams, seed: int = 42):
        self.params = params
        self.rng = np.random.default_rng(seed)
        
        # Regime parameters
        self.sigma_low = params.sigma * 0.6
        self.sigma_high = params.sigma * 2.0
        self.p_low_to_high = 0.005    # Transition probability per step
        self.p_high_to_low = 0.02
        
        # State
        self.price = params.initial_price
        self.regime = 'low'  # Start in low-vol regime
        self.price_history = [self.price]
        self.regime_history = ['low']
    
    def step(self) -> Tuple[float, str]:
        """Advance mid-price by one step."""
        # Regime transition
        if self.regime == 'low':
            if self.rng.random() < self.p_low_to_high:
                self.regime = 'high'
        else:
            if self.rng.random() < self.p_high_to_low:
                self.regime = 'low'
        
        # Volatility
        sigma = self.sigma_low if self.regime == 'low' else self.sigma_high
        
        # Drift (mean-reversion in low vol, momentum in high vol)
        if self.regime == 'low':
            drift = -0.001 * (self.price - self.params.initial_price)
        else:
            # Momentum: continue recent direction
            if len(self.price_history) > 5:
                recent_return = self.price - self.price_history[-5]
                drift = 0.01 * np.sign(recent_return)
            else:
                drift = 0.0
        
        # Update
        dW = self.rng.standard_normal()
        self.price += drift + sigma * dW
        self.price = max(self.price, self.params.tick_size)
        
        self.price_history.append(self.price)
        self.regime_history.append(self.regime)
        
        return self.price, self.regime


# ============================================================================
# Avellaneda-Stoikov Quoter
# ============================================================================

class AvellanedaStoikovQuoter:
    """
    Optimal quoting strategy from Avellaneda & Stoikov (2008).
    
    The reservation price adjusts for inventory risk:
        r = s - q * gamma * sigma^2 * (T - t)
    
    The optimal spread incorporates risk aversion and arrival intensity:
        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
    """
    
    def __init__(self, strategy: MMStrategy):
        self.strategy = strategy
    
    def compute_quotes(
        self,
        mid_price: float,
        inventory: int,
        sigma: float,
        time_remaining: float
    ) -> Tuple[float, float]:
        """
        Compute optimal bid and ask prices.
        
        Returns:
            (bid_price, ask_price)
        """
        s = self.strategy
        
        # Reservation price (inventory-adjusted fair value)
        reservation = mid_price - inventory * s.gamma * sigma**2 * time_remaining
        
        # Optimal spread
        spread = s.gamma * sigma**2 * time_remaining + (2 / s.gamma) * np.log(1 + s.gamma / s.k)
        half_spread = spread / 2
        
        # Clip spread
        half_spread = np.clip(half_spread, s.min_spread * mid_price, s.max_spread * mid_price)
        
        # Additional inventory skew
        skew = inventory * s.skew_factor
        
        bid_price = reservation - half_spread - skew
        ask_price = reservation + half_spread - skew
        
        # Ensure bid < ask and positive
        bid_price = max(bid_price, mid_price * 0.95)
        ask_price = max(ask_price, bid_price + 2 * mid_price * s.min_spread)
        
        # Round to tick size
        tick = 0.01
        bid_price = np.floor(bid_price / tick) * tick
        ask_price = np.ceil(ask_price / tick) * tick
        
        return bid_price, ask_price


# ============================================================================
# Order Flow Generator
# ============================================================================

class OrderFlowGenerator:
    """
    Generates incoming order flow with informed and uninformed components.
    
    Uninformed flow: Poisson arrival, random side
    Informed flow: directional, anticipating future mid-price moves
    """
    
    def __init__(self, params: MarketParams, seed: int = 123):
        self.params = params
        self.rng = np.random.default_rng(seed)
    
    def generate_orders(
        self,
        mid_price: float,
        future_direction: float = 0.0
    ) -> List[Tuple[str, float, int]]:
        """
        Generate incoming orders for one time step.
        
        Returns:
            List of (side, price, size) tuples
        """
        orders = []
        p = self.params
        
        # Uninformed flow
        n_buy = self.rng.poisson(p.lambda_buy * (1 - p.informed_fraction))
        n_sell = self.rng.poisson(p.lambda_sell * (1 - p.informed_fraction))
        
        for _ in range(n_buy):
            size = max(10, int(self.rng.normal(p.order_size_mean, p.order_size_std)))
            # Uninformed: willing to cross spread
            price = mid_price + self.rng.exponential(0.005) * mid_price
            orders.append(('buy', price, size))
        
        for _ in range(n_sell):
            size = max(10, int(self.rng.normal(p.order_size_mean, p.order_size_std)))
            price = mid_price - self.rng.exponential(0.005) * mid_price
            orders.append(('sell', price, size))
        
        # Informed flow (directional)
        n_informed = self.rng.poisson((p.lambda_buy + p.lambda_sell) / 2 * p.informed_fraction)
        for _ in range(n_informed):
            size = max(10, int(self.rng.normal(p.order_size_mean * 1.5, p.order_size_std)))
            if future_direction > 0:
                # Price going up: informed buys aggressively
                price = mid_price + p.informed_edge * mid_price
                orders.append(('buy', price, size))
            elif future_direction < 0:
                price = mid_price - p.informed_edge * mid_price
                orders.append(('sell', price, size))
            else:
                # No edge: random
                side = 'buy' if self.rng.random() > 0.5 else 'sell'
                price = mid_price + (1 if side == 'buy' else -1) * self.rng.exponential(0.003) * mid_price
                orders.append((side, price, size))
        
        return orders


# ============================================================================
# Simulation Engine
# ============================================================================

@dataclass
class SimulationState:
    """Running state of the simulation."""
    timestamp: int = 0
    inventory: int = 0
    cash: float = 0.0
    n_fills_buy: int = 0
    n_fills_sell: int = 0
    total_volume: int = 0
    
    # Tracking
    pnl_history: list = field(default_factory=list)
    inventory_history: list = field(default_factory=list)
    mid_history: list = field(default_factory=list)
    spread_history: list = field(default_factory=list)
    regime_history: list = field(default_factory=list)
    bid_history: list = field(default_factory=list)
    ask_history: list = field(default_factory=list)
    
    @property
    def mark_to_market(self) -> float:
        """Current P&L including inventory mark."""
        if self.mid_history:
            return self.cash + self.inventory * self.mid_history[-1]
        return self.cash


class MarketMakingSimulator:
    """
    Full market-making simulation engine.
    
    Runs the simulation loop:
    1. Mid-price evolves (with regime switching)
    2. Quoter computes optimal bid/ask
    3. Order flow arrives
    4. Fills are matched
    5. State is updated
    """
    
    def __init__(
        self,
        market_params: MarketParams = None,
        strategy: MMStrategy = None,
        seed: int = 42
    ):
        self.market_params = market_params or MarketParams()
        self.strategy = strategy or MMStrategy()
        
        self.mid_process = MidPriceProcess(self.market_params, seed=seed)
        self.quoter = AvellanedaStoikovQuoter(self.strategy)
        self.order_flow = OrderFlowGenerator(self.market_params, seed=seed + 1)
        self.book = OrderBook()
        self.state = SimulationState()
    
    def run(self, n_steps: int = 23400) -> SimulationState:
        """
        Run simulation for n_steps (default: 1 trading day = 390 min * 60 sec = 23400).
        
        For faster simulation, use n_steps = 2340 (1 step per 10 seconds).
        """
        state = self.state
        
        for step in range(n_steps):
            state.timestamp = step
            
            # 1. Mid-price evolution
            mid, regime = self.mid_process.step()
            
            # Estimate current volatility (rolling window)
            if len(self.mid_process.price_history) > 20:
                recent = np.array(self.mid_process.price_history[-20:])
                returns = np.diff(recent) / recent[:-1]
                sigma_est = np.std(returns) * np.sqrt(len(returns))
            else:
                sigma_est = self.market_params.sigma
            
            # Time remaining in session (fraction)
            time_remaining = max((n_steps - step) / n_steps * self.strategy.T, 1e-6)
            
            # 2. Compute quotes
            bid, ask = self.quoter.compute_quotes(
                mid, state.inventory, sigma_est, time_remaining
            )
            
            # Inventory limits: widen/pull quotes
            if abs(state.inventory) >= self.strategy.max_inventory:
                if state.inventory > 0:
                    # Long: aggressive ask, pull bid
                    ask = mid - 0.001 * mid
                    bid = mid - 0.05 * mid  # Pull bid far away
                else:
                    bid = mid + 0.001 * mid
                    ask = mid + 0.05 * mid
            
            # 3. Place quotes
            self.book.place_quotes(bid, ask, self.strategy.order_size, step)
            
            # 4. Generate and match order flow
            future_dir = 0
            if len(self.mid_process.price_history) > 2:
                future_dir = np.sign(self.mid_process.price_history[-1] - self.mid_process.price_history[-2])
            
            incoming = self.order_flow.generate_orders(mid, future_dir)
            
            for side, price, size in incoming:
                fills = self.book.check_fills(
                    price, side, size, mid, step, state.inventory
                )
                
                for fill in fills:
                    if fill.side == 'buy':
                        state.cash -= fill.price * fill.size
                        state.inventory += fill.size
                        state.n_fills_buy += 1
                    else:
                        state.cash += fill.price * fill.size
                        state.inventory -= fill.size
                        state.n_fills_sell += 1
                    state.total_volume += fill.size
            
            # 5. Record state
            state.pnl_history.append(state.mark_to_market)
            state.inventory_history.append(state.inventory)
            state.mid_history.append(mid)
            state.spread_history.append(ask - bid)
            state.regime_history.append(regime)
            state.bid_history.append(bid)
            state.ask_history.append(ask)
        
        return state


# ============================================================================
# Analytics
# ============================================================================

class PerformanceAnalytics:
    """P&L analytics for the market-making simulation."""
    
    def __init__(self, state: SimulationState):
        self.state = state
        self.pnl = np.array(state.pnl_history)
        self.inventory = np.array(state.inventory_history)
        self.mid = np.array(state.mid_history)
        self.spreads = np.array(state.spread_history)
        self.regimes = np.array(state.regime_history)
    
    def summary(self) -> dict:
        """Compute performance summary statistics."""
        pnl = self.pnl
        returns = np.diff(pnl)
        
        total_pnl = pnl[-1]
        
        # Sharpe (annualised, assuming 252 trading days)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * len(returns))
        else:
            sharpe = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(pnl)
        drawdown = pnl - peak
        max_dd = np.min(drawdown)
        
        # Fill rate
        total_fills = self.state.n_fills_buy + self.state.n_fills_sell
        
        # Average spread
        avg_spread = np.mean(self.spreads)
        avg_spread_bps = avg_spread / np.mean(self.mid) * 10000
        
        # Inventory stats
        avg_inv = np.mean(np.abs(self.inventory))
        max_inv = np.max(np.abs(self.inventory))
        
        # P&L by regime
        pnl_by_regime = {}
        for regime in ['low', 'high']:
            mask = self.regimes == regime
            if np.sum(mask) > 1:
                regime_pnl = np.diff(pnl[:-1][mask[:-1]])  
                pnl_by_regime[regime] = {
                    'total': np.sum(regime_pnl) if len(regime_pnl) > 0 else 0,
                    'mean': np.mean(regime_pnl) if len(regime_pnl) > 0 else 0,
                    'std': np.std(regime_pnl) if len(regime_pnl) > 0 else 0,
                    'steps': int(np.sum(mask)),
                    'pct_time': np.mean(mask),
                }
        
        # Spread earned vs inventory cost
        spread_pnl = 0
        inv_pnl = 0
        for fill in self.state.book.fills if hasattr(self.state, 'book') else []:
            pass
        
        return {
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_fills': total_fills,
            'buy_fills': self.state.n_fills_buy,
            'sell_fills': self.state.n_fills_sell,
            'total_volume': self.state.total_volume,
            'avg_spread_bps': avg_spread_bps,
            'avg_inventory': avg_inv,
            'max_inventory': max_inv,
            'final_inventory': self.state.inventory,
            'pnl_by_regime': pnl_by_regime,
        }
    
    def decompose_pnl(self) -> dict:
        """
        Decompose P&L into spread capture and inventory carry.
        
        Spread capture: profit from bid-ask spread
        Inventory carry: P&L from holding inventory as mid moves
        """
        fills = self.state.book.fills if hasattr(self.state, 'book') else []
        
        spread_pnl = 0.0
        for fill in fills:
            if fill.side == 'buy':
                spread_pnl += (fill.mid_at_fill - fill.price) * fill.size
            else:
                spread_pnl += (fill.price - fill.mid_at_fill) * fill.size
        
        total_pnl = self.pnl[-1]
        inventory_pnl = total_pnl - spread_pnl
        
        return {
            'total_pnl': total_pnl,
            'spread_pnl': spread_pnl,
            'inventory_pnl': inventory_pnl,
            'spread_pct': spread_pnl / max(abs(total_pnl), 1),
        }


# ============================================================================
# Multi-Day Simulation
# ============================================================================

def run_multi_day(
    n_days: int = 20,
    steps_per_day: int = 2340,
    market_params: MarketParams = None,
    strategy: MMStrategy = None,
    seed: int = 42
) -> dict:
    """
    Run multiple independent trading days and aggregate results.
    
    Returns daily P&L series and aggregate statistics.
    """
    daily_pnl = []
    daily_volume = []
    daily_fills = []
    daily_max_inv = []
    daily_sharpe = []
    all_states = []
    
    for day in range(n_days):
        sim = MarketMakingSimulator(
            market_params=market_params,
            strategy=strategy,
            seed=seed + day * 100
        )
        state = sim.run(n_steps=steps_per_day)
        analytics = PerformanceAnalytics(state)
        summary = analytics.summary()
        
        daily_pnl.append(summary['total_pnl'])
        daily_volume.append(summary['total_volume'])
        daily_fills.append(summary['total_fills'])
        daily_max_inv.append(summary['max_inventory'])
        all_states.append(state)
    
    daily_pnl = np.array(daily_pnl)
    
    # Aggregate stats
    return {
        'daily_pnl': daily_pnl,
        'cumulative_pnl': np.cumsum(daily_pnl),
        'mean_daily_pnl': np.mean(daily_pnl),
        'std_daily_pnl': np.std(daily_pnl),
        'sharpe_ratio': np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252) if np.std(daily_pnl) > 0 else 0,
        'win_rate': np.mean(daily_pnl > 0),
        'max_daily_loss': np.min(daily_pnl),
        'max_daily_gain': np.max(daily_pnl),
        'avg_volume': np.mean(daily_volume),
        'avg_fills': np.mean(daily_fills),
        'avg_max_inventory': np.mean(daily_max_inv),
        'n_days': n_days,
        'states': all_states,
    }


# ============================================================================
# Parameter Sensitivity
# ============================================================================

def gamma_sensitivity(
    gammas: np.ndarray = None,
    n_days: int = 10,
    steps_per_day: int = 2340
) -> dict:
    """Analyse P&L sensitivity to risk aversion parameter gamma."""
    if gammas is None:
        gammas = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    
    results = {}
    for g in gammas:
        strategy = MMStrategy(gamma=g)
        multi = run_multi_day(n_days=n_days, steps_per_day=steps_per_day, strategy=strategy)
        results[g] = {
            'mean_pnl': multi['mean_daily_pnl'],
            'std_pnl': multi['std_daily_pnl'],
            'sharpe': multi['sharpe_ratio'],
            'win_rate': multi['win_rate'],
            'avg_max_inv': multi['avg_max_inventory'],
        }
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("MARKET-MAKING SIMULATOR")
    print("Avellaneda-Stoikov Optimal Quoting Framework")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 60)
    
    # ── Single day simulation ──
    print("\n--- Single Day Simulation ---")
    params = MarketParams()
    strategy = MMStrategy(gamma=0.01, k=1.5, max_inventory=500)
    
    t0 = time.time()
    sim = MarketMakingSimulator(params, strategy, seed=42)
    state = sim.run(n_steps=2340)  # 1 step per 10 seconds
    t_elapsed = time.time() - t0
    
    analytics = PerformanceAnalytics(state)
    # Store book reference for PnL decomposition
    state.book = sim.book
    analytics = PerformanceAnalytics(state)
    summary = analytics.summary()
    decomp = analytics.decompose_pnl()
    
    print(f"Simulation time: {t_elapsed:.2f}s ({2340} steps)")
    print(f"\nP&L SUMMARY")
    print(f"{'='*40}")
    print(f"Total P&L:        ${summary['total_pnl']:,.2f}")
    print(f"Sharpe ratio:     {summary['sharpe_ratio']:.2f}")
    print(f"Max drawdown:     ${summary['max_drawdown']:,.2f}")
    print(f"Total fills:      {summary['total_fills']}")
    print(f"  Buy fills:      {summary['buy_fills']}")
    print(f"  Sell fills:     {summary['sell_fills']}")
    print(f"Total volume:     {summary['total_volume']:,}")
    print(f"Avg spread:       {summary['avg_spread_bps']:.1f} bps")
    print(f"Avg |inventory|:  {summary['avg_inventory']:.0f}")
    print(f"Max |inventory|:  {summary['max_inventory']}")
    print(f"Final inventory:  {summary['final_inventory']}")
    
    print(f"\nP&L DECOMPOSITION")
    print(f"{'='*40}")
    print(f"Spread capture:   ${decomp['spread_pnl']:,.2f}")
    print(f"Inventory carry:  ${decomp['inventory_pnl']:,.2f}")
    print(f"Spread % of P&L:  {decomp['spread_pct']:.1%}")
    
    print(f"\nP&L BY REGIME")
    print(f"{'='*40}")
    for regime, stats in summary['pnl_by_regime'].items():
        print(f"{regime.upper()} VOL: {stats['pct_time']:.0%} of time, P&L=${stats['total']:.2f}")
    
    # ── Multi-day simulation ──
    print(f"\n\n--- Multi-Day Simulation (20 days) ---")
    t0 = time.time()
    multi = run_multi_day(n_days=20, steps_per_day=2340, seed=42)
    t_elapsed = time.time() - t0
    
    print(f"Simulation time: {t_elapsed:.2f}s")
    print(f"\nAGGREGATE RESULTS")
    print(f"{'='*40}")
    print(f"Mean daily P&L:   ${multi['mean_daily_pnl']:,.2f}")
    print(f"Std daily P&L:    ${multi['std_daily_pnl']:,.2f}")
    print(f"Sharpe ratio:     {multi['sharpe_ratio']:.2f}")
    print(f"Win rate:         {multi['win_rate']:.0%}")
    print(f"Max daily loss:   ${multi['max_daily_loss']:,.2f}")
    print(f"Max daily gain:   ${multi['max_daily_gain']:,.2f}")
    print(f"Avg volume/day:   {multi['avg_volume']:,.0f}")
    print(f"Cumulative P&L:   ${multi['cumulative_pnl'][-1]:,.2f}")
    
    # ── Gamma sensitivity ──
    print(f"\n\n--- Gamma Sensitivity ---")
    print(f"{'Gamma':<10} {'Mean P&L':>12} {'Std P&L':>12} {'Sharpe':>8} {'Win%':>8} {'Max Inv':>8}")
    print("-" * 62)
    
    gamma_results = gamma_sensitivity(n_days=10, steps_per_day=2340)
    for g, stats in gamma_results.items():
        print(f"{g:<10.3f} ${stats['mean_pnl']:>11,.2f} ${stats['std_pnl']:>11,.2f} {stats['sharpe']:>7.2f} {stats['win_rate']:>7.0%} {stats['avg_max_inv']:>7.0f}")
    
    return state, multi, gamma_results


if __name__ == "__main__":
    state, multi, gamma_results = main()
