"""
FX Options Pricing Engine
===========================
Complete FX derivatives framework: Garman-Kohlhagen, Dupire local volatility,
Bergomi local-stochastic volatility, barrier options, Greeks surface.

Models:
  - Garman-Kohlhagen: closed-form European FX options (call/put)
  - Dupire Local Volatility: surface construction via forward PDE inversion
  - Bergomi LSV: local vol × stochastic vol factor (OU process for log-vol)
  - Barrier Options: MC pricing under local vol and LSV

Pipeline:
  1. Build implied vol surface from market quotes (or synthetic)
  2. Construct Dupire local vol surface via finite differences
  3. Price vanillas: GK closed-form, MC local vol, PDE local vol
  4. Price exotics: barriers under local vol and LSV
  5. Greeks: Delta, Gamma, Vega, Theta, Rho via bump-and-revalue

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import time


# ============================================================================
# FX Market Data
# ============================================================================

@dataclass
class FXMarket:
    """FX market data."""
    spot: float = 1.25           # EUR/USD spot
    rd: float = 0.02             # Domestic (USD) rate
    rf: float = 0.01             # Foreign (EUR) rate
    
    @property
    def forward(self) -> float:
        return self.spot * np.exp((self.rd - self.rf))
    
    def forward_at(self, T: float) -> float:
        return self.spot * np.exp((self.rd - self.rf) * T)


# ============================================================================
# Garman-Kohlhagen (GK) Model
# ============================================================================

def gk_call(S0, K, T, rd, rf, sigma):
    """Garman-Kohlhagen call price."""
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)


def gk_put(S0, K, T, rd, rf, sigma):
    """Garman-Kohlhagen put price via put-call parity."""
    call = gk_call(S0, K, T, rd, rf, sigma)
    return call + K * np.exp(-rd * T) - S0 * np.exp(-rf * T)


def gk_price(S0, K, T, rd, rf, sigma, option_type='call'):
    """Unified GK pricing."""
    if option_type == 'call':
        return gk_call(S0, K, T, rd, rf, sigma)
    return gk_put(S0, K, T, rd, rf, sigma)


def implied_vol(price, S0, K, T, rd, rf, option_type='call'):
    """Implied volatility inversion via Brentq."""
    if price <= 0 or T <= 0:
        return np.nan
    def obj(sigma):
        return gk_price(S0, K, T, rd, rf, sigma, option_type) - price
    try:
        return brentq(obj, 1e-4, 5.0)
    except ValueError:
        return np.nan


# ============================================================================
# Greeks (GK closed-form)
# ============================================================================

class GKGreeks:
    """Analytical Greeks for GK model."""
    
    @staticmethod
    def delta(S0, K, T, rd, rf, sigma, option_type='call'):
        d1 = (np.log(S0 / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return np.exp(-rf * T) * norm.cdf(d1)
        return np.exp(-rf * T) * (norm.cdf(d1) - 1)
    
    @staticmethod
    def gamma(S0, K, T, rd, rf, sigma):
        d1 = (np.log(S0 / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-rf * T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S0, K, T, rd, rf, sigma):
        d1 = (np.log(S0 / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S0 * np.exp(-rf * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol
    
    @staticmethod
    def theta(S0, K, T, rd, rf, sigma, option_type='call'):
        d1 = (np.log(S0 / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        first = -S0 * np.exp(-rf * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            return (first + rf * S0 * np.exp(-rf * T) * norm.cdf(d1) 
                    - rd * K * np.exp(-rd * T) * norm.cdf(d2)) / 252
        return (first - rf * S0 * np.exp(-rf * T) * norm.cdf(-d1)
                + rd * K * np.exp(-rd * T) * norm.cdf(-d2)) / 252
    
    @staticmethod
    def rho_domestic(S0, K, T, rd, rf, sigma, option_type='call'):
        d2 = (np.log(S0 / K) + (rd - rf - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return K * T * np.exp(-rd * T) * norm.cdf(d2) / 100
        return -K * T * np.exp(-rd * T) * norm.cdf(-d2) / 100


# ============================================================================
# Dupire Local Volatility Surface
# ============================================================================

class DupireLocalVol:
    """
    Dupire local volatility surface from option prices.
    
    sigma_loc^2(T,K) = [dC/dT + (rd-rf)*K*dC/dK + rf*C] / [0.5*K^2*d2C/dK2]
    """
    
    def __init__(self, maturities, strikes, call_prices, rd, rf):
        self.maturities = maturities
        self.strikes = strikes
        self.call_prices = call_prices
        self.rd = rd
        self.rf = rf
        self.interp = self._build()
    
    def _build(self) -> RegularGridInterpolator:
        nT, nK = self.call_prices.shape
        dT = np.diff(self.maturities)
        dK = self.strikes[1] - self.strikes[0]
        
        local_vol2 = np.zeros_like(self.call_prices)
        
        for i in range(nT):
            for j in range(nK):
                T, K, C = self.maturities[i], self.strikes[j], self.call_prices[i, j]
                
                # dC/dT
                if i == 0:
                    dC_dT = (self.call_prices[i+1, j] - C) / dT[0]
                elif i == nT - 1:
                    dC_dT = (C - self.call_prices[i-1, j]) / dT[-1]
                else:
                    dC_dT = (self.call_prices[i+1, j] - self.call_prices[i-1, j]) / (self.maturities[i+1] - self.maturities[i-1])
                
                # dC/dK
                if j == 0:
                    dC_dK = (self.call_prices[i, j+1] - C) / dK
                elif j == nK - 1:
                    dC_dK = (C - self.call_prices[i, j-1]) / dK
                else:
                    dC_dK = (self.call_prices[i, j+1] - self.call_prices[i, j-1]) / (2 * dK)
                
                # d2C/dK2
                if j == 0:
                    d2C = (self.call_prices[i, j] - 2*self.call_prices[i, j+1] + self.call_prices[i, j+2]) / dK**2
                elif j == nK - 1:
                    d2C = (self.call_prices[i, j-2] - 2*self.call_prices[i, j-1] + C) / dK**2
                else:
                    d2C = (self.call_prices[i, j+1] - 2*C + self.call_prices[i, j-1]) / dK**2
                
                num = dC_dT + (self.rd - self.rf) * K * dC_dK + self.rf * C
                den = 0.5 * K**2 * d2C
                
                local_vol2[i, j] = num / den if den > 0 and num > 0 else np.nan
        
        local_vol = np.sqrt(np.maximum(np.nan_to_num(local_vol2, nan=0.04), 1e-6))
        
        return RegularGridInterpolator(
            (self.maturities, self.strikes), local_vol,
            method='linear', bounds_error=False, fill_value=None
        )
    
    def __call__(self, t, S):
        """Evaluate local vol at (t, S)."""
        pts = np.atleast_2d(np.column_stack((np.atleast_1d(t), np.atleast_1d(S))))
        return self.interp(pts)


# ============================================================================
# Bergomi Local-Stochastic Volatility (LSV)
# ============================================================================

@dataclass
class BergomiParams:
    """Bergomi LSV parameters."""
    kappa: float = 1.5       # Mean reversion of vol factor
    theta: float = 0.04      # Long-run variance
    eta: float = 0.5         # Vol-of-vol
    rho: float = -0.4        # Spot-vol correlation
    v0: float = 0.04         # Initial variance


class BergomiLSVEngine:
    """
    Bergomi-style local-stochastic volatility simulation.
    
    dS/S = (rd - rf) dt + sigma_loc(t,S) * sqrt(V) * dW1
    dV   = kappa*(theta - V) dt + eta * sqrt(V) * dW2
    corr(dW1, dW2) = rho
    """
    
    def __init__(self, params: BergomiParams = None, seed: int = 42):
        self.p = params or BergomiParams()
        self.rng = np.random.default_rng(seed)
    
    def simulate(
        self,
        S0: float,
        rd: float,
        rf: float,
        local_vol_func,
        T: float,
        n_paths: int = 50000,
        n_steps: int = 200,
        barrier: float = None,
        barrier_type: str = 'down-and-out'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate LSV paths.
        
        Returns:
            S_final: terminal spot values
            alive: boolean mask (for barriers)
            paths: (n_paths, n_steps+1) full paths if needed
        """
        p = self.p
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.full(n_paths, S0)
        V = np.full(n_paths, p.v0)
        alive = np.ones(n_paths, dtype=bool)
        
        for step in range(n_steps):
            t = step * dt
            
            dW1 = self.rng.standard_normal(n_paths) * sqrt_dt
            dW_indep = self.rng.standard_normal(n_paths) * sqrt_dt
            dW2 = p.rho * dW1 + np.sqrt(1 - p.rho**2) * dW_indep
            
            # Local vol evaluation
            pts = np.column_stack((np.full(n_paths, t), np.clip(S, 0.5 * S0, 2.0 * S0)))
            try:
                sigma_loc = local_vol_func(pts).flatten()
            except Exception:
                sigma_loc = np.full(n_paths, 0.20)
            
            sqrt_V = np.sqrt(np.maximum(V, 0))
            sigma_inst = sigma_loc * sqrt_V
            
            # Spot dynamics
            S = S * np.exp((rd - rf - 0.5 * sigma_inst**2) * dt + sigma_inst * dW1)
            S = np.maximum(S, 1e-8)
            
            # Variance dynamics
            V = np.abs(V + p.kappa * (p.theta - V) * dt + p.eta * sqrt_V * dW2)
            
            # Barrier check
            if barrier is not None:
                if barrier_type == 'down-and-out':
                    alive &= (S > barrier)
                elif barrier_type == 'up-and-out':
                    alive &= (S < barrier)
        
        return S, alive, V


# ============================================================================
# MC Pricer (Local Vol)
# ============================================================================

def mc_local_vol(S0, K, T, rd, rf, local_vol_func, n_paths=50000, n_steps=200, option_type='call'):
    """Price European option under local vol via MC."""
    rng = np.random.default_rng(42)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    
    for step in range(n_steps):
        t = step * dt
        Z = rng.standard_normal(n_paths)
        pts = np.column_stack((np.full(n_paths, t), S))
        sigma = local_vol_func(pts).flatten()
        S = S * np.exp((rd - rf - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S = np.maximum(S, 1e-8)
    
    if option_type == 'call':
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    
    return np.exp(-rd * T) * np.mean(payoff)


# ============================================================================
# PDE Pricer (Local Vol)
# ============================================================================

def pde_local_vol(S0, K, T, rd, rf, local_vol_func, nS=200, nT=200, option_type='call'):
    """Price European option under local vol via implicit FD."""
    S_max = 3 * S0
    S_grid = np.linspace(0, S_max, nS)
    dS = S_grid[1] - S_grid[0]
    dt = T / nT
    
    if option_type == 'call':
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)
    
    for step in range(nT, 0, -1):
        t = (step - 1) * dt
        V_old = V.copy()
        
        if option_type == 'call':
            V[0] = 0.0
            V[-1] = S_grid[-1] - K * np.exp(-rd * (T - t))
        else:
            V[0] = K * np.exp(-rd * (T - t))
            V[-1] = 0.0
        
        sigma_vals = local_vol_func(np.column_stack((np.full(nS, t), S_grid))).flatten()
        a = 0.5 * sigma_vals**2 * S_grid**2
        b = (rd - rf) * S_grid
        
        n = nS - 2
        A = np.zeros(n)
        B = np.zeros(n)
        C = np.zeros(n)
        D = np.zeros(n)
        
        for i in range(1, nS - 1):
            idx = i - 1
            A[idx] = -dt * (a[i] / dS**2 - b[i] / (2 * dS))
            B[idx] = 1 + 2 * dt * a[i] / dS**2 + dt * rd
            C[idx] = -dt * (a[i] / dS**2 + b[i] / (2 * dS))
            D[idx] = V_old[i]
        
        D[0] -= A[0] * V[0]
        D[-1] -= C[-1] * V[-1]
        
        # Thomas algorithm
        c_p = np.zeros(n)
        d_p = np.zeros(n)
        c_p[0] = C[0] / B[0]
        d_p[0] = D[0] / B[0]
        for i in range(1, n):
            denom = B[i] - A[i] * c_p[i-1]
            c_p[i] = C[i] / denom if i < n-1 else 0
            d_p[i] = (D[i] - A[i] * d_p[i-1]) / denom
        
        V_new = np.zeros(n)
        V_new[-1] = d_p[-1]
        for i in range(n-2, -1, -1):
            V_new[i] = d_p[i] - c_p[i] * V_new[i+1]
        V[1:-1] = V_new
    
    return np.interp(S0, S_grid, V)


# ============================================================================
# Barrier Pricing (Local Vol + LSV)
# ============================================================================

def barrier_mc(S0, K, T, rd, rf, local_vol_func, barrier, barrier_type='down-and-out',
               option_type='call', n_paths=100000, n_steps=500):
    """Price barrier option under local vol via MC."""
    rng = np.random.default_rng(42)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    alive = np.ones(n_paths, dtype=bool)
    
    for step in range(n_steps):
        t = step * dt
        Z = rng.standard_normal(n_paths)
        pts = np.column_stack((np.full(n_paths, t), np.clip(S, 0.1, 5*S0)))
        sigma = local_vol_func(pts).flatten()
        S = S * np.exp((rd - rf - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S = np.maximum(S, 1e-8)
        
        if barrier_type == 'down-and-out':
            alive &= (S > barrier)
        elif barrier_type == 'up-and-out':
            alive &= (S < barrier)
        elif barrier_type == 'down-and-in':
            alive |= (S <= barrier)  # Knock-in: set alive when hit
        elif barrier_type == 'up-and-in':
            alive |= (S >= barrier)
    
    if 'out' in barrier_type:
        mask = alive
    else:  # knock-in
        mask = alive  # For KI, alive means barrier was hit at some point
    
    if option_type == 'call':
        payoff = np.where(mask, np.maximum(S - K, 0), 0)
    else:
        payoff = np.where(mask, np.maximum(K - S, 0), 0)
    
    return np.exp(-rd * T) * np.mean(payoff)


# ============================================================================
# Smile Builder (Synthetic)
# ============================================================================

def build_smile_market(S0=1.25, rd=0.02, rf=0.01, base_vol=0.20, skew=-0.03, convexity=0.02):
    """
    Build a synthetic FX smile (not flat).
    
    IV(K) = base_vol + skew * log(K/F) + convexity * log(K/F)^2
    """
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    moneyness = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
    strikes = S0 * moneyness
    
    call_prices = np.zeros((len(maturities), len(strikes)))
    iv_surface = np.zeros_like(call_prices)
    
    for i, T in enumerate(maturities):
        F = S0 * np.exp((rd - rf) * T)
        for j, K in enumerate(strikes):
            log_m = np.log(K / F)
            iv = base_vol + skew * log_m + convexity * log_m**2
            iv = max(iv, 0.05)
            iv_surface[i, j] = iv
            call_prices[i, j] = gk_call(S0, K, T, rd, rf, iv)
    
    return maturities, strikes, call_prices, iv_surface


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("FX OPTIONS PRICING ENGINE")
    print("GK / Dupire Local Vol / Bergomi LSV / Barriers / Greeks")
    print("Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE")
    print("=" * 65)
    
    S0, rd, rf = 1.25, 0.02, 0.01
    market = FXMarket(S0, rd, rf)
    
    # Build smile market
    maturities, strikes, call_prices, iv_surface = build_smile_market(S0, rd, rf)
    
    print(f"\nMarket: EUR/USD spot={S0}, rd={rd:.1%}, rf={rf:.1%}")
    print(f"Smile: {len(maturities)} maturities × {len(strikes)} strikes")
    
    # Dupire local vol
    dupire = DupireLocalVol(maturities, strikes, call_prices, rd, rf)
    
    # Price 1Y ATM call: GK vs MC vs PDE
    K_atm = S0
    T = 1.0
    iv_atm = iv_surface[2, 4]  # 1Y ATM
    
    t0 = time.time()
    gk = gk_call(S0, K_atm, T, rd, rf, iv_atm)
    mc_lv = mc_local_vol(S0, K_atm, T, rd, rf, dupire.interp, n_paths=50000, n_steps=200)
    pde_lv = pde_local_vol(S0, K_atm, T, rd, rf, dupire.interp)
    
    # Bergomi LSV
    bergomi = BergomiLSVEngine(BergomiParams(), seed=42)
    S_final, alive, _ = bergomi.simulate(S0, rd, rf, dupire.interp, T, n_paths=50000, n_steps=200)
    lsv_price = np.exp(-rd * T) * np.mean(np.maximum(S_final - K_atm, 0))
    
    print(f"\n1Y ATM CALL PRICING (IV={iv_atm:.1%})")
    print(f"{'='*40}")
    print(f"GK closed-form:  {gk:.6f}")
    print(f"MC Local Vol:    {mc_lv:.6f}")
    print(f"PDE Local Vol:   {pde_lv:.6f}")
    print(f"Bergomi LSV:     {lsv_price:.6f}")
    print(f"Time: {time.time()-t0:.2f}s")
    
    # Greeks
    greeks = GKGreeks()
    print(f"\nGREEKS (1Y ATM Call)")
    print(f"Delta:  {greeks.delta(S0, K_atm, T, rd, rf, iv_atm):>8.4f}")
    print(f"Gamma:  {greeks.gamma(S0, K_atm, T, rd, rf, iv_atm):>8.4f}")
    print(f"Vega:   {greeks.vega(S0, K_atm, T, rd, rf, iv_atm):>8.4f}")
    print(f"Theta:  {greeks.theta(S0, K_atm, T, rd, rf, iv_atm):>8.6f}")
    print(f"Rho:    {greeks.rho_domestic(S0, K_atm, T, rd, rf, iv_atm):>8.4f}")
    
    # Barrier option
    barrier = 0.90 * S0
    bar_lv = barrier_mc(S0, K_atm, T, rd, rf, dupire.interp, barrier, 'down-and-out', n_paths=100000, n_steps=300)
    
    S_bar, alive_bar, _ = bergomi.simulate(S0, rd, rf, dupire.interp, T, n_paths=100000, n_steps=300, barrier=barrier)
    bar_lsv = np.exp(-rd * T) * np.mean(np.where(alive_bar, np.maximum(S_bar - K_atm, 0), 0))
    
    print(f"\nBARRIER (Down-and-Out Call, B={barrier:.3f})")
    print(f"Local Vol MC:   {bar_lv:.6f}")
    print(f"Bergomi LSV:    {bar_lsv:.6f}")
    print(f"Vanilla - Barrier: {gk - bar_lv:.6f} (barrier discount)")
    
    # IV surface
    print(f"\nIMPLIED VOL SURFACE")
    print(f"{'Strike':<8}" + "".join(f"  {T:.2f}Y" for T in maturities))
    for j, K in enumerate(strikes):
        row = f"{K:.3f}  "
        for i in range(len(maturities)):
            row += f" {iv_surface[i,j]:>5.1%}"
        print(row)
    
    return market, dupire, iv_surface, maturities, strikes


if __name__ == "__main__":
    main()
