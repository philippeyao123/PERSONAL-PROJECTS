"""
Module 3 — Microstructure Feature Engineering
===============================================
Computes four canonical market microstructure signals:

1. VPIN — Volume-Synchronized Probability of Informed Trading (Easley et al., 2012)
2. Order Flow Imbalance (OFI) — signed volume pressure
3. Kyle's Lambda — price impact coefficient (Kyle, 1985)
4. Realized Volatility — per-window, annualized

References:
  Easley, D., Lopez de Prado, M.M. & O'Hara, M. (2012). Flow Toxicity and
  Liquidity in a High-Frequency World. Review of Financial Studies, 25(5).

  Kyle, A.S. (1985). Continuous Auctions and Insider Trading.
  Econometrica, 53(6), 1315–1335.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ============================================================================
# VPIN — Volume-Synchronized Probability of Informed Trading
# ============================================================================

def compute_vpin(
    df: pd.DataFrame,
    bucket_size: Optional[int] = None,
    n_buckets: int = 50
) -> pd.Series:
    """
    VPIN estimator (Easley, Lopez de Prado & O'Hara, 2012).

    Algorithm:
    1. Sort trades by time
    2. Group into volume buckets of size V = total_volume / n_buckets
    3. Within each bucket, classify volume as buy (V_b) or sell (V_s)
       using bulk volume classification (BVC) or Lee-Ready
    4. VPIN_n = (1/tau) * sum_{i=n-tau+1}^{n} |V_b^i - V_s^i| / V
       where tau = lookback window (default: n_buckets // 2)

    High VPIN → elevated informed trading → wider spreads expected.

    Parameters
    ----------
    df : DataFrame with trade_size, lr_direction
    bucket_size : volume per bucket (auto-computed if None)
    n_buckets : number of buckets to target

    Returns
    -------
    Series of VPIN per original row (forward-filled from bucket boundaries)
    """
    df = df.copy().reset_index(drop=True)

    total_vol = df['trade_size'].sum()
    if bucket_size is None:
        bucket_size = max(1, total_vol // n_buckets)

    # Assign bucket IDs by cumulative volume
    cum_vol = df['trade_size'].cumsum()
    df['vpin_bucket'] = (cum_vol / bucket_size).astype(int)

    # Aggregate per bucket
    agg = df.groupby('vpin_bucket').apply(lambda g: pd.Series({
        'buy_vol':  (g.loc[g['lr_direction'] ==  1, 'trade_size']).sum(),
        'sell_vol': (g.loc[g['lr_direction'] == -1, 'trade_size']).sum(),
        'total_vol': g['trade_size'].sum(),
    })).reset_index()

    agg['imbalance'] = np.abs(agg['buy_vol'] - agg['sell_vol'])

    # Rolling VPIN over tau buckets
    tau = max(2, len(agg) // 4)
    agg['vpin'] = (
        agg['imbalance'].rolling(tau, min_periods=1).sum() /
        (tau * bucket_size)
    ).clip(0, 1)

    # Map back to original DataFrame index
    bucket_to_vpin = dict(zip(agg['vpin_bucket'], agg['vpin']))
    result = df['vpin_bucket'].map(bucket_to_vpin)
    return result.rename('vpin')


# ============================================================================
# Order Flow Imbalance (OFI)
# ============================================================================

def compute_ofi(
    df: pd.DataFrame,
    window_seconds: int = 60
) -> pd.Series:
    """
    Order Flow Imbalance (Cont, Kukanov & Stoikov, 2014).

    OFI_t = (buy_volume - sell_volume) / total_volume in window [t-w, t]

    Ranges from -1 (pure selling pressure) to +1 (pure buying pressure).
    Strong predictor of short-horizon price moves.
    """
    df = df.copy()
    df['signed_vol'] = df['trade_size'] * df['lr_direction']
    df['bucket'] = df['timestamp'].dt.floor(f'{window_seconds}s')

    agg = df.groupby('bucket').agg(
        buy_vol=('trade_size', lambda x: x[df.loc[x.index, 'lr_direction'] == 1].sum()),
        sell_vol=('trade_size', lambda x: x[df.loc[x.index, 'lr_direction'] == -1].sum()),
    )
    agg['total_vol'] = agg['buy_vol'] + agg['sell_vol']
    agg['ofi'] = np.where(
        agg['total_vol'] > 0,
        (agg['buy_vol'] - agg['sell_vol']) / agg['total_vol'],
        0.0
    )

    bucket_to_ofi = dict(zip(agg.index, agg['ofi']))
    return df['bucket'].map(bucket_to_ofi).rename('ofi')


# ============================================================================
# Kyle's Lambda — Price Impact
# ============================================================================

def compute_kyle_lambda(
    df: pd.DataFrame,
    window_seconds: int = 300
) -> pd.Series:
    """
    Kyle's Lambda: price impact coefficient from OLS regression.

        Δp_t = λ * x_t + ε_t

    where x_t = signed order flow (buy volume - sell volume).
    λ > 0 → price rises with net buying pressure.
    Larger λ → less liquid, more informed flow.

    Parameters
    ----------
    window_seconds : rolling window for regression
    """
    df = df.copy()
    df['bucket'] = df['timestamp'].dt.floor('10s')
    df['price_change'] = df.groupby('bucket')['trade_price'].transform(
        lambda x: x.iloc[-1] - x.iloc[0]
    )
    df['signed_vol'] = df['trade_size'] * df['lr_direction']

    agg = df.groupby('bucket').agg(
        price_change=('price_change', 'last'),
        net_flow=('signed_vol', 'sum'),
    ).dropna()

    # Rolling OLS (λ = cov(Δp, x) / var(x))
    window_bins = max(2, window_seconds // 10)
    cov_px = agg['price_change'].rolling(window_bins).cov(agg['net_flow'])
    var_x = agg['net_flow'].rolling(window_bins).var()
    agg['kyle_lambda'] = (cov_px / var_x.replace(0, np.nan)).fillna(0).clip(lower=0)

    bucket_to_lambda = dict(zip(agg.index, agg['kyle_lambda']))
    df['bucket'] = df['timestamp'].dt.floor('10s')
    return df['bucket'].map(bucket_to_lambda).rename('kyle_lambda')


# ============================================================================
# Realized Volatility
# ============================================================================

def compute_realized_vol(
    df: pd.DataFrame,
    window_seconds: int = 300,
    annualize: bool = True,
    trading_seconds: int = 23400   # 6.5h trading day
) -> pd.Series:
    """
    Realized volatility via sum of squared log-returns within window.

        RV_t = sqrt( sum_{i in window} (ln(p_i / p_{i-1}))^2 )

    Optionally annualized: RV_annual = RV_window * sqrt(trading_seconds / window)
    """
    df = df.copy()
    df['log_ret'] = np.log(df['trade_price'] / df['trade_price'].shift(1))
    df['log_ret2'] = df['log_ret'] ** 2

    # Use time-based rolling window (approximate by obs count)
    avg_trades_per_sec = len(df) / (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    window_obs = max(2, int(window_seconds * avg_trades_per_sec))

    rv = np.sqrt(df['log_ret2'].rolling(window_obs, min_periods=2).sum())

    if annualize:
        scale = np.sqrt(trading_seconds / window_seconds)
        rv = rv * scale

    return rv.rename('realized_vol')


# ============================================================================
# Build Feature Matrix
# ============================================================================

def build_feature_matrix(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """
    Assemble all microstructure features into a single DataFrame.

    Returns
    -------
    DataFrame with: timestamp, mid, spread (true), vpin, ofi,
                    kyle_lambda, realized_vol
    """
    feats = df[['timestamp', 'mid', 'spread', 'vol_regime']].copy()

    feats['vpin']         = compute_vpin(df).values
    feats['ofi']          = compute_ofi(df, window_seconds).values
    feats['kyle_lambda']  = compute_kyle_lambda(df).values
    feats['realized_vol'] = compute_realized_vol(df, window_seconds).values

    # Lag features (t-1) to avoid look-ahead
    for col in ['vpin', 'ofi', 'kyle_lambda', 'realized_vol']:
        feats[f'{col}_lag1'] = feats[col].shift(1)

    feats = feats.dropna().reset_index(drop=True)
    return feats


if __name__ == '__main__':
    from module1_data import simulate_tick_data, lee_ready_classify, MarketConfig

    cfg = MarketConfig()
    df = simulate_tick_data(cfg)
    df['lr_direction'] = lee_ready_classify(df)

    feats = build_feature_matrix(df)
    print(f"Feature matrix shape: {feats.shape}")
    print(f"\nCorrelations with true spread:")
    cols = ['vpin', 'ofi', 'kyle_lambda', 'realized_vol']
    for c in cols:
        corr = feats[[c, 'spread']].dropna().corr().iloc[0, 1]
        print(f"  {c:<20}: {corr:+.3f}")
    print(f"\n{feats[['timestamp','spread','vpin','ofi','kyle_lambda','realized_vol']].head(8).round(4).to_string()}")
