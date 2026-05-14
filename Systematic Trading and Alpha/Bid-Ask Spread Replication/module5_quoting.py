"""
Module 5 — Optimal Quoting & Backtest (Avellaneda-Stoikov)
============================================================
Uses the predicted spread from Module 4 as a volatility proxy input to the
Avellaneda-Stoikov (2008) optimal quoting model.

The market-maker solves a stochastic control problem:
    Maximize E[W_T - φ * q_T²]
    subject to inventory dynamics and Poisson order arrivals

Optimal quotes:
    r(s, q, t) = s - q * γ * σ² * (T - t)        [reservation price]
    δ_a = δ_b = γ * σ² * (T - t) / 2 + (1/γ) * ln(1 + γ/k)  [half-spread]

    bid_t = r_t - δ_b
    ask_t = r_t + δ_a

Order arrival (symmetric Poisson):
    λ_a(δ) = A * exp(-k * δ)  [ask fill rate]
    λ_b(δ) = A * exp(-k * δ)  [bid fill rate]

Risk-adjusted with inventory skew: market-maker skews quotes when inventory
accumulates to reduce adverse selection exposure.

References:
    Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit
    order book. Quantitative Finance, 8(3), 217–224.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ============================================================================
# Model Parameters
# ============================================================================

@dataclass
class ASParams:
    """Avellaneda-Stoikov model parameters."""
    gamma: float = 0.1          # Risk aversion coefficient
    k: float = 1.5              # Order arrival decay (fill probability sensitivity)
    A: float = 1.0              # Baseline order arrival rate
    T: float = 1.0              # Horizon (normalized)
    max_inventory: int = 50     # Hard inventory cap
    min_spread_bps: float = 0.5 # Floor on quoted spread (50bp on spread, not price)
    vol_scale: float = 1.0      # Multiplier for predicted vol input


# ============================================================================
# Avellaneda-Stoikov Quoter
# ============================================================================

class ASQuoter:
    """
    Implements A-S optimal quoting with predicted spread as volatility proxy.

    Key insight: the predicted bid-ask spread (from Module 4) is a real-time
    estimate of σ² * (T-t). We use it directly rather than computing realized vol,
    making the pipeline fully data-driven.
    """

    def __init__(self, params: ASParams = ASParams()):
        self.p = params

    def sigma_from_spread(self, predicted_spread: float, t: float) -> float:
        """
        Back out implied σ from predicted spread.
        spread_AS ≈ γ * σ² * (T - t)  [dominant term for large T-t]
        σ² ≈ spread / (γ * (T - t))
        """
        tau = max(self.p.T - t, 0.001)
        sigma2 = predicted_spread * self.p.vol_scale / (self.p.gamma * tau + 1e-8)
        return np.sqrt(max(sigma2, 0))

    def reservation_price(
        self, s: float, q: float, t: float, sigma: float
    ) -> float:
        """r(s, q, t) = s - q * γ * σ² * (T - t)"""
        tau = max(self.p.T - t, 0.001)
        return s - q * self.p.gamma * sigma ** 2 * tau

    def optimal_half_spread(self, t: float, sigma: float) -> float:
        """
        δ* = γ * σ² * (T-t) / 2 + (1/γ) * ln(1 + γ/k)
        """
        tau = max(self.p.T - t, 0.001)
        term1 = self.p.gamma * sigma ** 2 * tau / 2
        term2 = (1 / self.p.gamma) * np.log(1 + self.p.gamma / self.p.k)
        delta = term1 + term2
        min_half = self.p.min_spread_bps / 2 / 100
        return max(delta, min_half)

    def quote(
        self,
        s: float,
        q: float,
        t: float,
        predicted_spread: float,
        vpin: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Compute optimal bid and ask prices.

        Adverse selection adjustment: when VPIN is elevated, widen spread
        asymmetrically to penalize the informed side.

        Parameters
        ----------
        s : current mid price
        q : current inventory (signed)
        t : normalized time ∈ [0, T]
        predicted_spread : spread forecast from Module 4
        vpin : VPIN estimate (0=uninformed, 1=fully informed)

        Returns
        -------
        (bid, ask) tuple
        """
        sigma = self.sigma_from_spread(predicted_spread, t)
        r = self.reservation_price(s, q, t, sigma)
        delta = self.optimal_half_spread(t, sigma)

        # VPIN-based adverse selection widening (max 2x base spread)
        tox_mult = 1 + vpin * 1.0
        delta_adj = delta * tox_mult

        # Inventory skew: reduce exposure on the side we're long
        # If q > 0 (long), we lower the ask to offload, raise the bid
        skew = np.sign(q) * min(abs(q) / (self.p.max_inventory + 1), 0.5) * delta

        bid = r - delta_adj + skew
        ask = r + delta_adj + skew

        return max(bid, 0.0), max(ask, bid + 1e-6)

    def fill_probability(self, delta: float) -> float:
        """P(fill | quoted half-spread δ) = exp(-k * δ)"""
        return np.exp(-self.p.k * delta)


# ============================================================================
# Backtest Engine
# ============================================================================

@dataclass
class BacktestState:
    cash: float = 0.0
    inventory: int = 0
    trades: List[Dict] = field(default_factory=list)
    quotes: List[Dict] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    predicted_spreads: np.ndarray,
    vpins: np.ndarray,
    params: ASParams = ASParams(),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Backtest the A-S market maker on simulated tick data.

    At each time step:
    1. Quoter posts optimal bid/ask using predicted spread + VPIN
    2. Incoming orders arrive (Poisson) and fill based on fill probability
    3. Inventory and cash updated
    4. MTM P&L = cash + inventory * mid

    Parameters
    ----------
    df : tick DataFrame (module1 output with lr_direction)
    predicted_spreads : per-row spread forecast (Module 4 output)
    vpins : per-row VPIN estimates (Module 3 output)

    Returns
    -------
    DataFrame with per-step P&L, inventory, quotes, and attributions
    """
    rng = np.random.default_rng(seed)
    quoter = ASQuoter(params)
    state = BacktestState()

    n = len(df)
    T = params.T
    results = []

    for i in range(n):
        row = df.iloc[i]
        t = i / n * T
        s = row['mid']
        pred_spread = float(predicted_spreads[i]) if i < len(predicted_spreads) else row['spread']
        vpin_val = float(vpins[i]) if i < len(vpins) else 0.5

        # Clamp inventory
        q = np.clip(state.inventory, -params.max_inventory, params.max_inventory)

        bid, ask = quoter.quote(s, q, t, pred_spread, vpin_val)
        half_spread = (ask - bid) / 2
        fill_prob = quoter.fill_probability(half_spread)

        # Simulate fills from incoming flow
        n_arrivals = rng.poisson(params.A)
        for _ in range(n_arrivals):
            direction = row['lr_direction']   # Use actual market order direction

            if direction == 1:    # Market buy → hits our ask
                if rng.random() < fill_prob:
                    size = max(1, int(rng.lognormal(4.0, 0.5)))
                    size = min(size, params.max_inventory - abs(state.inventory))
                    if size > 0:
                        state.cash += ask * size
                        state.inventory -= size
                        state.trades.append({
                            'step': i, 'side': 'sell_to_buyer',
                            'price': ask, 'size': size,
                        })

            elif direction == -1: # Market sell → hits our bid
                if rng.random() < fill_prob:
                    size = max(1, int(rng.lognormal(4.0, 0.5)))
                    size = min(size, params.max_inventory - abs(state.inventory))
                    if size > 0:
                        state.cash -= bid * size
                        state.inventory += size
                        state.trades.append({
                            'step': i, 'side': 'buy_from_seller',
                            'price': bid, 'size': size,
                        })

        # Mark-to-market
        mtm = state.cash + state.inventory * s
        realized_pnl = state.cash + state.inventory * s

        # P&L Attribution
        spread_income = sum(
            (t['price'] - s) * t['size'] * (1 if t['side'] == 'sell_to_buyer' else -1)
            for t in state.trades[-n_arrivals:]
        ) if state.trades else 0.0
        inventory_pnl = state.inventory * (s - df.iloc[max(0, i-1)]['mid'])

        results.append({
            'step': i,
            'timestamp': row['timestamp'],
            'mid': s,
            'bid': bid,
            'ask': ask,
            'quoted_spread': ask - bid,
            'predicted_spread': pred_spread,
            'true_spread': row['spread'],
            'vpin': vpin_val,
            'inventory': state.inventory,
            'cash': state.cash,
            'mtm_pnl': mtm,
            'spread_income': spread_income,
            'inventory_pnl': inventory_pnl,
            'vol_regime': row['vol_regime'],
            'n_fills': len([t for t in state.trades if t['step'] == i]),
        })

    return pd.DataFrame(results)


def summarise_backtest(bt: pd.DataFrame) -> Dict:
    """Compute key performance metrics from backtest results."""
    pnl = bt['mtm_pnl']
    returns = pnl.diff().dropna()

    sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252 * 3600)
    max_dd = (pnl - pnl.cummax()).min()
    total_fills = bt['n_fills'].sum()
    avg_inv = bt['inventory'].abs().mean()
    max_inv = bt['inventory'].abs().max()

    # Regime breakdown
    regime_pnl = bt.groupby('vol_regime')['mtm_pnl'].last() - bt.groupby('vol_regime')['mtm_pnl'].first()

    return {
        'total_pnl': pnl.iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_fills': total_fills,
        'avg_abs_inventory': avg_inv,
        'max_abs_inventory': max_inv,
        'spread_income_total': bt['spread_income'].sum(),
        'inventory_pnl_total': bt['inventory_pnl'].sum(),
        'regime_pnl': regime_pnl.to_dict(),
    }


if __name__ == '__main__':
    from module1_data import simulate_tick_data, lee_ready_classify, MarketConfig
    from module2_reconstruction import level2_proxy
    from module3_features import build_feature_matrix, compute_vpin
    from module4_modelling import train_spread_models

    cfg = MarketConfig()
    df = simulate_tick_data(cfg)
    df['lr_direction'] = lee_ready_classify(df)
    feats = build_feature_matrix(df)
    l2 = level2_proxy(df).set_index('bucket_time')['spread_proxy']

    res = train_spread_models(feats, l2)

    # Use GBM predictions (best model) as spread input
    # Align predictions back to tick-level (forward fill from 30s buckets)
    gbm_preds_ts = pd.Series(
        res.predictions['GBM'],
        index=feats.groupby(feats['timestamp'].dt.floor('30s'))['timestamp'].first().iloc[int(len(feats)*0.8):]
    )
    pred_per_tick = df['timestamp'].map(
        lambda t: res.predictions['GBM'][0]  # simplified alignment for demo
    ).fillna(df['spread'].mean())

    vpin_vals = build_feature_matrix(df)['vpin'].reindex(df.index).fillna(0.5)

    bt = run_backtest(
        df.head(5000),
        predicted_spreads=pred_per_tick.head(5000).values,
        vpins=vpin_vals.head(5000).values,
    )
    summary = summarise_backtest(bt)

    print("Backtest Summary:")
    print(f"  Total P&L         : {summary['total_pnl']:+.2f}")
    print(f"  Sharpe Ratio      : {summary['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown      : {summary['max_drawdown']:.2f}")
    print(f"  Total Fills       : {summary['total_fills']:,}")
    print(f"  Avg |Inventory|   : {summary['avg_abs_inventory']:.1f}")
    print(f"  Spread Income     : {summary['spread_income_total']:.2f}")
