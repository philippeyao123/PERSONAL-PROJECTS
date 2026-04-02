"""
Extension 2: Adverse Selection & Flow Toxicity (PIN Model)
============================================================
Estimates the probability of informed trading (PIN) in real-time
and dynamically adjusts spreads.

PIN Model (Easley, Kiefer, O'Hara & Paperman, 1996):
    PIN = alpha * mu / (alpha * mu + epsilon_b + epsilon_s)

where:
    alpha   = probability of information event
    mu      = informed trader arrival rate
    epsilon = uninformed arrival rates (buy/sell)

The market-maker estimates PIN from observed order flow imbalance
and widens spreads when toxicity is high.

Also implements VPIN (Volume-synchronized PIN) for real-time estimation.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PINParams:
    """PIN model parameters."""
    alpha: float = 0.15       # Probability of information event
    mu: float = 50.0          # Informed arrival rate
    epsilon_b: float = 100.0  # Uninformed buy arrival rate
    epsilon_s: float = 100.0  # Uninformed sell arrival rate
    
    @property
    def pin(self) -> float:
        """Probability of Informed Trading."""
        return self.alpha * self.mu / (self.alpha * self.mu + self.epsilon_b + self.epsilon_s)
    
    @property
    def expected_spread_component(self) -> float:
        """Adverse selection component of the spread."""
        return self.pin  # Proportional to PIN


@dataclass  
class FlowMetrics:
    """Real-time order flow metrics."""
    buy_volume: int = 0
    sell_volume: int = 0
    n_trades: int = 0
    
    @property
    def total_volume(self) -> int:
        return self.buy_volume + self.sell_volume
    
    @property
    def order_imbalance(self) -> float:
        """Signed order imbalance [-1, 1]."""
        total = self.total_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total
    
    @property
    def abs_imbalance(self) -> float:
        return abs(self.order_imbalance)


class VPINEstimator:
    """
    Volume-synchronized Probability of Informed Trading.
    
    VPIN buckets trades by volume (not time) and measures the
    imbalance within each bucket. High VPIN indicates toxic flow.
    
    VPIN = sum(|V_buy - V_sell|) / (n * V_bucket)
    """
    
    def __init__(self, bucket_size: int = 1000, n_buckets: int = 50):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        
        self.current_buy = 0
        self.current_sell = 0
        self.current_volume = 0
        
        self.bucket_imbalances: List[float] = []
    
    def add_trade(self, side: str, size: int):
        """Add a trade to the current bucket."""
        if side == 'buy':
            self.current_buy += size
        else:
            self.current_sell += size
        self.current_volume += size
        
        # Check if bucket is full
        if self.current_volume >= self.bucket_size:
            imbalance = abs(self.current_buy - self.current_sell)
            self.bucket_imbalances.append(imbalance)
            
            # Reset bucket
            self.current_buy = 0
            self.current_sell = 0
            self.current_volume = 0
            
            # Keep only recent buckets
            if len(self.bucket_imbalances) > self.n_buckets * 2:
                self.bucket_imbalances = self.bucket_imbalances[-self.n_buckets:]
    
    @property
    def vpin(self) -> float:
        """Current VPIN estimate."""
        if len(self.bucket_imbalances) < 5:
            return 0.15  # Default prior
        
        recent = self.bucket_imbalances[-self.n_buckets:]
        return np.mean(recent) / self.bucket_size
    
    @property
    def vpin_percentile(self) -> float:
        """VPIN as percentile of historical distribution."""
        if len(self.bucket_imbalances) < 20:
            return 0.5
        
        current = self.vpin
        historical = [
            np.mean(self.bucket_imbalances[max(0,i-self.n_buckets):i]) / self.bucket_size
            for i in range(self.n_buckets, len(self.bucket_imbalances))
        ]
        if len(historical) == 0:
            return 0.5
        return np.mean(np.array(historical) <= current)


class ToxicityAwareQuoter:
    """
    Market-maker that adjusts spreads based on real-time flow toxicity.
    
    Spread = base_spread + toxicity_premium
    
    When VPIN is high:
    - Widen spreads to compensate for adverse selection
    - Reduce quote size
    - Skew quotes more aggressively away from toxic side
    """
    
    def __init__(
        self,
        base_gamma: float = 0.01,
        toxicity_multiplier: float = 3.0,
        vpin_threshold: float = 0.3
    ):
        self.base_gamma = base_gamma
        self.toxicity_multiplier = toxicity_multiplier
        self.vpin_threshold = vpin_threshold
    
    def adjusted_spread(
        self,
        mid_price: float,
        base_half_spread: float,
        vpin: float,
        order_imbalance: float
    ) -> Tuple[float, float]:
        """
        Compute toxicity-adjusted bid and ask.
        
        Returns:
            (bid, ask) adjusted for flow toxicity
        """
        # Toxicity premium: exponential increase above threshold
        if vpin > self.vpin_threshold:
            excess = (vpin - self.vpin_threshold) / (1 - self.vpin_threshold)
            premium = base_half_spread * self.toxicity_multiplier * excess
        else:
            premium = 0.0
        
        adjusted_half = base_half_spread + premium
        
        # Skew based on order imbalance (lean away from toxic side)
        # If imbalance > 0 (more buying), widen ask, tighten bid
        skew = order_imbalance * adjusted_half * 0.5
        
        bid = mid_price - adjusted_half + skew
        ask = mid_price + adjusted_half + skew
        
        return bid, ask
    
    def adjusted_size(self, base_size: int, vpin: float) -> int:
        """Reduce quote size when flow is toxic."""
        if vpin > self.vpin_threshold:
            reduction = (vpin - self.vpin_threshold) / (1 - self.vpin_threshold)
            return max(10, int(base_size * (1 - 0.7 * reduction)))
        return base_size


class AdverseSelectionSimulator:
    """
    Simulation with dynamic adverse selection measurement.
    
    Generates order flow with time-varying informed fraction
    and measures how well the MM adapts.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.vpin_estimator = VPINEstimator(bucket_size=500, n_buckets=30)
        self.quoter = ToxicityAwareQuoter()
    
    def run(self, n_steps: int = 2340) -> dict:
        """Run simulation with dynamic adverse selection."""
        
        mid = 100.0
        sigma = 0.02
        inventory = 0
        cash = 0.0
        
        # Time-varying toxicity (information events)
        toxicity_schedule = np.zeros(n_steps)
        # Create information events at random times
        event_starts = self.rng.choice(n_steps, size=5, replace=False)
        for start in event_starts:
            end = min(start + self.rng.integers(50, 200), n_steps)
            toxicity_schedule[start:end] = self.rng.uniform(0.3, 0.6)
        
        # Tracking
        history = {
            'mid': [], 'bid': [], 'ask': [], 'pnl': [],
            'inventory': [], 'vpin': [], 'true_toxicity': [],
            'spread_bps': [], 'imbalance': []
        }
        
        flow_window = FlowMetrics()
        window_size = 100
        buy_window = []
        sell_window = []
        
        for step in range(n_steps):
            true_tox = toxicity_schedule[step]
            
            # Mid-price with drift from informed flow
            if true_tox > 0:
                direction = 1 if self.rng.random() > 0.5 else -1
                mid += direction * true_tox * sigma + sigma * self.rng.standard_normal()
            else:
                mid += sigma * self.rng.standard_normal()
            mid = max(mid, 1.0)
            
            # Base spread from AS
            tau = max((n_steps - step) / n_steps * (1/252), 1e-8)
            base_half = 0.01 * sigma**2 * tau * 10000 + 0.001 * mid
            
            # Get VPIN and flow metrics
            vpin = self.vpin_estimator.vpin
            
            # Rolling imbalance
            if len(buy_window) > window_size:
                buy_window = buy_window[-window_size:]
                sell_window = sell_window[-window_size:]
            total_buy = sum(buy_window)
            total_sell = sum(sell_window)
            total = total_buy + total_sell
            imbalance = (total_buy - total_sell) / max(total, 1)
            
            # Toxicity-adjusted quotes
            bid, ask = self.quoter.adjusted_spread(mid, base_half, vpin, imbalance)
            
            # Generate flow
            informed_frac = true_tox
            n_orders = self.rng.poisson(10)
            
            step_buys = 0
            step_sells = 0
            
            for _ in range(n_orders):
                is_informed = self.rng.random() < informed_frac
                
                if is_informed:
                    # Informed: trade in direction of future price move
                    side = 'buy' if mid > 100 else 'sell'
                    size = max(10, int(self.rng.normal(150, 30)))
                else:
                    side = 'buy' if self.rng.random() > 0.5 else 'sell'
                    size = max(10, int(self.rng.normal(100, 30)))
                
                self.vpin_estimator.add_trade(side, size)
                
                if side == 'buy':
                    step_buys += size
                    if ask > 0:
                        aggressor_px = mid + self.rng.exponential(0.003) * mid
                        if aggressor_px >= ask:
                            cash += ask * size
                            inventory -= size
                else:
                    step_sells += size
                    if bid > 0:
                        aggressor_px = mid - self.rng.exponential(0.003) * mid
                        if aggressor_px <= bid:
                            cash -= bid * size
                            inventory += size
            
            buy_window.append(step_buys)
            sell_window.append(step_sells)
            
            mtm = cash + inventory * mid
            spread_bps = (ask - bid) / mid * 10000
            
            history['mid'].append(mid)
            history['bid'].append(bid)
            history['ask'].append(ask)
            history['pnl'].append(mtm)
            history['inventory'].append(inventory)
            history['vpin'].append(vpin)
            history['true_toxicity'].append(true_tox)
            history['spread_bps'].append(spread_bps)
            history['imbalance'].append(imbalance)
        
        # Convert to arrays
        for k in history:
            history[k] = np.array(history[k])
        
        return history


if __name__ == "__main__":
    print("Adverse Selection & Flow Toxicity Extension")
    print("=" * 50)
    
    # PIN model
    pin_params = PINParams(alpha=0.15, mu=50, epsilon_b=100, epsilon_s=100)
    print(f"PIN = {pin_params.pin:.2%}")
    
    # Simulation
    sim = AdverseSelectionSimulator(seed=42)
    t0 = time.time()
    history = sim.run(n_steps=2340)
    print(f"Simulation: {time.time()-t0:.2f}s")
    
    print(f"\nRESULTS")
    print(f"Final P&L:     ${history['pnl'][-1]:,.2f}")
    print(f"Avg VPIN:      {np.mean(history['vpin']):.3f}")
    print(f"Max VPIN:      {np.max(history['vpin']):.3f}")
    print(f"Avg spread:    {np.mean(history['spread_bps']):.1f} bps")
    
    # Spread during toxic vs clean periods
    toxic_mask = history['true_toxicity'] > 0.1
    clean_mask = ~toxic_mask
    print(f"\nSpread during toxic flow:  {np.mean(history['spread_bps'][toxic_mask]):.1f} bps")
    print(f"Spread during clean flow: {np.mean(history['spread_bps'][clean_mask]):.1f} bps")
    print(f"Toxic periods: {np.mean(toxic_mask):.1%} of time")
