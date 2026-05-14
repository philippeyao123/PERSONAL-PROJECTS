"""
Module 1 — Data Ingestion & Tick Classification
================================================
Generates realistic synthetic tick data and classifies trades using the
Lee-Ready (1991) algorithm.

Lee-Ready Rules (in order):
  1. Quote rule: compare trade price to prevailing quote midpoint
     - trade > mid → buyer-initiated (+1)
     - trade < mid → seller-initiated (-1)
  2. Tick rule (fallback when trade = mid):
     - uptick / zero-uptick → buyer-initiated
     - downtick / zero-downtick → seller-initiated

Reference:
  Lee, C.M.C. & Ready, M.J. (1991). Inferring Trade Direction from Intraday Data.
  Journal of Finance, 46(2), 733–746.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# Synthetic Market Data Generator
# ============================================================================

@dataclass
class MarketConfig:
    """Configuration for synthetic market simulation."""
    S0: float = 100.0          # Initial mid price
    sigma: float = 0.002       # Per-step volatility (approx 3% daily)
    lambda_trade: float = 10.0 # Poisson arrival rate (trades/second)
    alpha: float = 0.25        # Fraction of informed traders
    base_spread: float = 0.04  # Base bid-ask spread (4 bps on S0=100)
    T_seconds: int = 3600      # Simulation horizon (1 hour)
    dt: float = 1.0            # Time step (seconds)
    seed: int = 42


def simulate_tick_data(cfg: MarketConfig = MarketConfig()) -> pd.DataFrame:
    """
    Simulate intraday tick data with realistic microstructure.

    Mid-price follows a GBM with informed-trader impact:
        dS_t = sigma * dW_t + alpha * I_t * sigma (information component)

    Spread varies with volatility regime:
        spread_t = base_spread * (1 + vol_multiplier_t)

    Returns
    -------
    DataFrame with columns:
        timestamp, mid, bid, ask, trade_price, trade_size,
        true_direction (ground truth for validation)
    """
    rng = np.random.default_rng(cfg.seed)
    n_steps = int(cfg.T_seconds / cfg.dt)

    # --- Mid price simulation (GBM with regime switching) ---
    vol_regimes = np.ones(n_steps)
    regime = 1.0
    for t in range(1, n_steps):
        if rng.random() < 0.002:               # Regime switch probability
            regime = rng.choice([0.5, 1.0, 2.0, 3.5])
        vol_regimes[t] = regime

    innovations = rng.standard_normal(n_steps)
    log_returns = cfg.sigma * vol_regimes * innovations
    mid_prices = cfg.S0 * np.exp(np.cumsum(log_returns))
    mid_prices = np.maximum(mid_prices, 1.0)   # Floor at 1

    # --- Dynamic spread (widens with volatility) ---
    spreads = cfg.base_spread * (0.5 + vol_regimes * 0.5 + 0.3 * np.abs(innovations))
    bids = mid_prices - spreads / 2
    asks = mid_prices + spreads / 2

    # --- Trade arrivals (Poisson process) ---
    n_trades_per_step = rng.poisson(cfg.lambda_trade * cfg.dt, size=n_steps)

    records = []
    for t in range(n_steps):
        if n_trades_per_step[t] == 0:
            continue

        for _ in range(n_trades_per_step[t]):
            # Informed or uninformed trader?
            is_informed = rng.random() < cfg.alpha

            if is_informed:
                # Informed: direction correlated with next mid move
                future_move = np.mean(log_returns[t:min(t+10, n_steps)])
                true_dir = 1 if future_move > 0 else -1
            else:
                # Uninformed: random direction
                true_dir = rng.choice([-1, 1])

            # Trade price: informed hit the quote, uninformed may trade inside
            noise = rng.normal(0, spreads[t] * 0.1)
            if true_dir == 1:   # Buy → hits ask
                trade_px = asks[t] + noise
            else:               # Sell → hits bid
                trade_px = bids[t] + noise

            trade_size = max(1, int(rng.lognormal(mean=4.5, sigma=0.8)))

            records.append({
                'timestamp': t,
                'mid': mid_prices[t],
                'bid': bids[t],
                'ask': asks[t],
                'spread': spreads[t],
                'vol_regime': vol_regimes[t],
                'trade_price': trade_px,
                'trade_size': trade_size,
                'true_direction': true_dir,
                'is_informed': is_informed,
            })

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', origin='2026-01-15')
    return df.reset_index(drop=True)


# ============================================================================
# Lee-Ready Trade Classification
# ============================================================================

def lee_ready_classify(df: pd.DataFrame) -> pd.Series:
    """
    Classify trade direction using the Lee-Ready (1991) algorithm.

    Operates on trade_price, bid, ask columns.
    Falls back to tick rule when trade is at midpoint.

    Returns
    -------
    Series of {+1, -1} (buyer/seller initiated)
    """
    directions = np.zeros(len(df), dtype=int)
    prices = df['trade_price'].values
    mids = ((df['bid'] + df['ask']) / 2).values

    for i in range(len(df)):
        p = prices[i]
        m = mids[i]

        # Quote rule
        if p > m + 1e-8:
            directions[i] = 1      # Above mid → buy
        elif p < m - 1e-8:
            directions[i] = -1     # Below mid → sell
        else:
            # Tick rule fallback
            if i == 0:
                directions[i] = 1
            else:
                prev_p = prices[i - 1]
                if p > prev_p:
                    directions[i] = 1
                elif p < prev_p:
                    directions[i] = -1
                else:
                    # Zero-tick: inherit last non-zero tick direction
                    directions[i] = directions[i - 1] if directions[i - 1] != 0 else 1

    return pd.Series(directions, index=df.index, name='lr_direction')


def classification_accuracy(df: pd.DataFrame) -> dict:
    """
    Compute Lee-Ready classification accuracy vs ground truth.
    Only meaningful on synthetic data where true_direction is known.
    """
    mask = df['true_direction'] != 0
    correct = (df.loc[mask, 'lr_direction'] == df.loc[mask, 'true_direction']).sum()
    total = mask.sum()
    informed_mask = mask & df['is_informed']
    uninformed_mask = mask & ~df['is_informed']

    return {
        'overall_accuracy': correct / total,
        'informed_accuracy': (
            df.loc[informed_mask, 'lr_direction'] == df.loc[informed_mask, 'true_direction']
        ).mean(),
        'uninformed_accuracy': (
            df.loc[uninformed_mask, 'lr_direction'] == df.loc[uninformed_mask, 'true_direction']
        ).mean(),
        'n_trades': total,
        'n_informed': informed_mask.sum(),
    }


if __name__ == '__main__':
    cfg = MarketConfig()
    df = simulate_tick_data(cfg)
    df['lr_direction'] = lee_ready_classify(df)
    acc = classification_accuracy(df)

    print(f"Trades generated : {len(df):,}")
    print(f"Overall accuracy : {acc['overall_accuracy']:.1%}")
    print(f"Informed accuracy : {acc['informed_accuracy']:.1%}")
    print(f"Uninformed accuracy: {acc['uninformed_accuracy']:.1%}")
    print(df[['timestamp', 'mid', 'bid', 'ask', 'trade_price',
               'lr_direction', 'true_direction']].head(10).to_string())
