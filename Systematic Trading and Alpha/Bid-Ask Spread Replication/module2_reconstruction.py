"""
Module 2 — Bid-Ask Spread Reconstruction
=========================================
Reconstructs bid-ask spreads from trades-only data using three estimators:

1. Roll (1984) — serial covariance of price changes
2. Corwin-Schultz (2012) — high-low range over consecutive periods
3. Level-2 Proxy — max(buy-side prices) / min(sell-side prices) per window

References:
  Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread
  in an Efficient Market. Journal of Finance, 39(4), 1127–1139.

  Corwin, S.A. & Schultz, P. (2012). A Simple Way to Estimate Bid-Ask Spreads
  from Daily High and Low Prices. Journal of Finance, 67(2), 719–760.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================================
# Roll (1984) Implicit Spread Estimator
# ============================================================================

def roll_spread(prices: pd.Series, window: int = 60) -> pd.Series:
    """
    Roll (1984) spread estimator.

    Model: price change = c * q_t + u_t
    where c = half-spread, q_t ∈ {±1} i.i.d., u_t = efficient price innovation

    The serial covariance of price changes:
        Cov(Δp_t, Δp_{t-1}) = -c²

    Hence: c = sqrt(-Cov) and spread = 2c

    When Cov > 0 (no bounce), set spread = 0 (Cov is biased by trends).

    Parameters
    ----------
    prices : trade price series
    window : rolling window in observations

    Returns
    -------
    Series of Roll spread estimates
    """
    delta_p = prices.diff()
    cov = delta_p.rolling(window).cov(delta_p.shift(1))

    # Roll spread: 2 * sqrt(-cov), floored at 0
    roll = 2 * np.sqrt(np.maximum(-cov, 0))
    return roll.rename('roll_spread')


# ============================================================================
# Corwin-Schultz (2012) High-Low Estimator
# ============================================================================

def corwin_schultz_spread(
    df: pd.DataFrame,
    price_col: str = 'trade_price',
    window_seconds: int = 60
) -> pd.Series:
    """
    Corwin-Schultz (2012) spread estimator using high-low price ranges.

    Theory: Over an interval [t, t+h], the high-low range reflects:
        - Volatility component: grows with sqrt(h)
        - Spread component: constant across intervals

    Let β = E[ln(H_t/L_t)²] and γ = E[ln(H_{t,t+1}/L_{t,t+1})²]

    Spread:
        α = (sqrt(2β) - sqrt(β)) / (3 - 2√2) - sqrt(γ / (3 - 2√2))
        S = 2(e^α - 1) / (1 + e^α)

    Floored at 0 (can go negative due to microstructure noise).

    Parameters
    ----------
    df : DataFrame with trade_price and timestamp
    window_seconds : aggregation window to compute H/L
    """
    df = df.copy()
    df['bucket'] = (df['timestamp'].astype(np.int64) // (window_seconds * 1_000_000_000))

    agg = df.groupby('bucket')['trade_price'].agg(['max', 'min']).reset_index()
    agg.columns = ['bucket', 'H', 'L']
    agg['H'] = np.maximum(agg['H'], agg['L'] + 1e-8)

    log_hl = np.log(agg['H'] / agg['L'])
    log_hl2 = log_hl ** 2

    # β: average of single-period squared log range
    beta = (log_hl2 + log_hl2.shift(1)) / 2

    # γ: squared log range over two consecutive periods
    H2 = np.maximum(agg['H'], agg['H'].shift(1))
    L2 = np.minimum(agg['L'], agg['L'].shift(1))
    gamma = np.log(H2 / L2) ** 2

    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = np.maximum(alpha, 0)

    cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    cs_spread = cs_spread.rename('cs_spread')

    # Map back to original df timestamps
    bucket_to_cs = dict(zip(agg['bucket'], cs_spread))
    return df['bucket'].map(bucket_to_cs).rename('cs_spread')


# ============================================================================
# Level-2 Proxy Reconstructor
# ============================================================================

def level2_proxy(
    df: pd.DataFrame,
    window_seconds: int = 30
) -> pd.DataFrame:
    """
    Reconstruct bid/ask from trades using Lee-Ready classification.

    Within each time window:
        bid_proxy  = max(trade_price where lr_direction == -1)   [best bid hit]
        ask_proxy  = min(trade_price where lr_direction == +1)   [best ask hit]
        spread     = ask_proxy - bid_proxy

    This is the Level-2 reconstruction from order flow, analogous to the
    method used in the Finalto e-trading test (Section 2).

    Parameters
    ----------
    df : DataFrame with trade_price, lr_direction, timestamp
    window_seconds : bin width

    Returns
    -------
    DataFrame with columns: bucket_time, bid_proxy, ask_proxy, spread_proxy, mid_proxy
    """
    df = df.copy()
    df['bucket'] = df['timestamp'].dt.floor(f'{window_seconds}s')

    buys  = df[df['lr_direction'] == 1].groupby('bucket')['trade_price'].max()
    sells = df[df['lr_direction'] == -1].groupby('bucket')['trade_price'].min()

    result = pd.DataFrame({'ask_proxy': buys, 'bid_proxy': sells}).dropna()
    result['spread_proxy'] = result['ask_proxy'] - result['bid_proxy']
    result['mid_proxy'] = (result['ask_proxy'] + result['bid_proxy']) / 2

    # Filter out inverted or zero spreads (microstructure noise)
    result = result[result['spread_proxy'] > 0].copy()
    result.index.name = 'bucket_time'
    return result.reset_index()


# ============================================================================
# Spread Comparison
# ============================================================================

def compare_estimators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three spread estimators and compare to true spread.

    Returns
    -------
    DataFrame with true_spread, roll_spread, cs_spread, l2_spread per bucket
    """
    # True spread (ground truth from simulation)
    df = df.copy()
    df['bucket'] = df['timestamp'].dt.floor('30s')
    true_spread = df.groupby('bucket')['spread'].mean()

    roll = roll_spread(df['trade_price'], window=60)
    df['roll_spread'] = roll.values

    df['cs_spread'] = corwin_schultz_spread(df, window_seconds=30).values

    l2 = level2_proxy(df, window_seconds=30).set_index('bucket_time')

    result = pd.DataFrame({
        'true_spread':  true_spread,
        'roll_spread':  df.groupby('bucket')['roll_spread'].mean(),
        'cs_spread':    df.groupby('bucket')['cs_spread'].mean(),
    }).dropna()

    # Join Level-2
    result = result.join(l2[['spread_proxy']].rename(columns={'spread_proxy': 'l2_spread'}), how='left')

    # Errors
    for col in ['roll_spread', 'cs_spread', 'l2_spread']:
        result[f'{col}_error'] = result[col] - result['true_spread']

    return result


if __name__ == '__main__':
    from module1_data import simulate_tick_data, lee_ready_classify, MarketConfig

    cfg = MarketConfig()
    df = simulate_tick_data(cfg)
    df['lr_direction'] = lee_ready_classify(df)

    comp = compare_estimators(df)

    print("Spread Estimator Comparison (mean absolute error vs true spread):")
    print("-" * 55)
    for est in ['roll_spread', 'cs_spread', 'l2_spread']:
        mae = comp[f'{est}_error'].abs().mean()
        corr = comp[[est, 'true_spread']].dropna().corr().iloc[0, 1]
        print(f"  {est:<15}: MAE = {mae:.4f}  |  Corr = {corr:.3f}")

    print(f"\nSample output:\n{comp[['true_spread','roll_spread','cs_spread','l2_spread']].head(10).round(4).to_string()}")
