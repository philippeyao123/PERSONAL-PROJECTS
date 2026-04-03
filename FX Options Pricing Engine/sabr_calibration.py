"""
Extension 1: SABR Smile Calibration
=====================================
SABR (Hagan et al., 2002) stochastic volatility model for FX smile.

    dF = sigma * F^beta * dW1
    dsigma = nu * sigma * dW2
    corr(dW1, dW2) = rho

The SABR model is the industry standard for FX options desks because:
1. It fits the smile with 4 intuitive parameters
2. Alpha controls ATM vol level
3. Beta controls backbone (CEV exponent)
4. Rho controls skew (negative = put skew)
5. Nu controls curvature (smile wings)

Calibration: fit (alpha, rho, nu) to market implied vols at fixed beta.

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Dict
import time


@dataclass
class SABRParams:
    """SABR model parameters."""
    alpha: float = 0.20     # Initial vol
    beta: float = 0.50      # CEV exponent (0=normal, 0.5=CIR-like, 1=lognormal)
    rho: float = -0.25      # Spot-vol correlation (negative = put skew for FX)
    nu: float = 0.40        # Vol-of-vol


def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan's SABR implied volatility approximation.
    
    Valid for moderate maturities and not-too-extreme strikes.
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return alpha
    
    # Handle ATM case
    if abs(F - K) < 1e-10:
        FK_mid = F
        log_FK = 0.0
        z = 0.0
        x_z = 1.0
    else:
        FK_mid = (F * K)**((1 - beta) / 2)
        log_FK = np.log(F / K)
        z = nu / alpha * FK_mid * log_FK
        
        # x(z) function
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho))
        if abs(x_z) < 1e-10:
            x_z = 1.0
        else:
            x_z = z / x_z
    
    # Correction terms
    FK_beta = (F * K)**((1 - beta) / 2)
    
    one_minus_beta = 1 - beta
    A = alpha / (FK_beta * (1 + one_minus_beta**2 / 24 * log_FK**2 + one_minus_beta**4 / 1920 * log_FK**4))
    
    B1 = one_minus_beta**2 / 24 * alpha**2 / FK_beta**2
    B2 = rho * beta * nu * alpha / (4 * FK_beta)
    B3 = (2 - 3 * rho**2) * nu**2 / 24
    B = 1 + (B1 + B2 + B3) * T
    
    return A * x_z * B


def sabr_smile(F, strikes, T, params: SABRParams):
    """Compute SABR implied vols for a range of strikes."""
    return np.array([
        sabr_implied_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu)
        for K in strikes
    ])


class SABRCalibrator:
    """
    Calibrate SABR parameters to market implied volatilities.
    
    Fixed beta (market convention for FX: typically 0.5 or 1.0).
    Calibrate (alpha, rho, nu) to minimise squared IV error.
    """
    
    def __init__(self, beta: float = 0.50):
        self.beta = beta
    
    def calibrate(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        market_ivs: np.ndarray,
        weights: np.ndarray = None
    ) -> SABRParams:
        """Calibrate SABR to market smile."""
        if weights is None:
            weights = np.ones(len(strikes))
        
        def objective(params):
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10
            
            model_ivs = np.array([
                sabr_implied_vol(F, K, T, alpha, self.beta, rho, nu)
                for K in strikes
            ])
            
            return np.sum(weights * (model_ivs - market_ivs)**2)
        
        # Initial guess from ATM vol
        atm_idx = np.argmin(np.abs(strikes - F))
        alpha0 = market_ivs[atm_idx]
        
        result = minimize(
            objective,
            x0=[alpha0, -0.2, 0.3],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8}
        )
        
        alpha, rho, nu = result.x
        rho = np.clip(rho, -0.999, 0.999)
        
        return SABRParams(alpha=alpha, beta=self.beta, rho=rho, nu=nu)
    
    def calibrate_term_structure(
        self,
        F_curve: np.ndarray,       # Forward at each maturity
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_iv_surface: np.ndarray  # (n_maturities, n_strikes)
    ) -> Dict[float, SABRParams]:
        """Calibrate SABR at each maturity (term structure of smile)."""
        results = {}
        for i, T in enumerate(maturities):
            F = F_curve[i]
            ivs = market_iv_surface[i]
            params = self.calibrate(F, strikes, T, ivs)
            results[T] = params
        return results


def demo_sabr_calibration():
    """Demonstrate SABR calibration on FX smile."""
    from fx_options_pricer import build_smile_market, FXMarket
    
    S0, rd, rf = 1.25, 0.02, 0.01
    maturities, strikes, call_prices, iv_surface = build_smile_market(
        S0, rd, rf, base_vol=0.20, skew=-0.03, convexity=0.02
    )
    
    print("SABR Smile Calibration")
    print("=" * 55)
    
    calibrator = SABRCalibrator(beta=0.50)
    
    print(f"\n{'Maturity':<10} {'Alpha':>8} {'Rho':>8} {'Nu':>8} {'RMSE(bps)':>10}")
    print("-" * 48)
    
    all_params = {}
    for i, T in enumerate(maturities):
        F = S0 * np.exp((rd - rf) * T)
        market_ivs = iv_surface[i]
        
        params = calibrator.calibrate(F, strikes, T, market_ivs)
        all_params[T] = params
        
        # Compute fit error
        model_ivs = sabr_smile(F, strikes, T, params)
        rmse = np.sqrt(np.mean((model_ivs - market_ivs)**2)) * 10000
        
        print(f"{T:<10.2f} {params.alpha:>7.3f} {params.rho:>+7.3f} {params.nu:>7.3f} {rmse:>9.1f}")
    
    # Show smile for 1Y
    T = 1.0
    F = S0 * np.exp((rd - rf) * T)
    params = all_params[T]
    
    dense_strikes = np.linspace(strikes[0], strikes[-1], 50)
    sabr_ivs = sabr_smile(F, dense_strikes, T, params)
    
    print(f"\n1Y SABR SMILE (α={params.alpha:.3f}, ρ={params.rho:+.3f}, ν={params.nu:.3f})")
    print(f"{'Strike':<10} {'SABR IV':>8} {'Market IV':>10}")
    print("-" * 30)
    for j, K in enumerate(strikes):
        sabr_iv = sabr_implied_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu)
        print(f"{K:<10.3f} {sabr_iv:>7.1%} {iv_surface[2,j]:>9.1%}")
    
    return all_params, maturities, strikes, iv_surface


if __name__ == "__main__":
    demo_sabr_calibration()
