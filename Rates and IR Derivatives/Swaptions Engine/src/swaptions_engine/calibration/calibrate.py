"""Calibrate G2++ to a swaption volatility surface.

Given a grid of co-terminal or standard ATM swaption normal vols, find the
G2++ parameters (a, sigma, b, eta, rho) that best fit the market prices in a
least-squares sense. The model European prices come from the semi-analytic
integral; market prices come from Bachelier with the quoted normal vol.

Calibration is the step that turns a textbook model into a desk tool: the
output parameters are what every downstream price and risk number depends on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from swaptions_engine.curve.yield_curve import Curve
from swaptions_engine.instruments.swap import VanillaSwap
from swaptions_engine.models.g2pp import G2pp, G2ppParams
from swaptions_engine.pricing.analytic import bachelier_swaption

logger = logging.getLogger(__name__)


@dataclass
class SurfacePoint:
    expiry: float
    tenor: float
    normal_vol: float  # quoted ATM normal (bp) vol, as a decimal (e.g. 0.006)


@dataclass
class CalibrationResult:
    params: G2ppParams
    rmse_vol: float           # RMSE in normal-vol terms (bp)
    fit_table: list[dict]     # per-point market vs model
    success: bool


class G2ppCalibrator:
    """Least-squares calibration of G2++ to ATM swaption normal vols."""

    def __init__(self, curve: Curve, freq: int = 2) -> None:
        self.curve = curve
        self.freq = freq

    def _market_price(self, pt: SurfacePoint) -> tuple[float, float, float]:
        sw0 = VanillaSwap(pt.expiry, pt.tenor, 0.0, self.freq)
        fwd = sw0.forward_swap_rate(self.curve)
        swap = VanillaSwap(pt.expiry, pt.tenor, fwd, self.freq)
        ann = swap.annuity(self.curve)
        price = bachelier_swaption(fwd, fwd, pt.expiry, pt.normal_vol, ann,
                                   payer=True)
        return price, fwd, ann

    def calibrate(
        self, surface: list[SurfacePoint],
        x0: G2ppParams | None = None,
    ) -> CalibrationResult:
        market = [self._market_price(pt) for pt in surface]

        # Parameter vector p = [a, sigma, b, eta, rho]; transform to enforce
        # bounds (a,b,sigma,eta > 0 ; rho in (-1,1)).
        if x0 is None:
            x0 = G2ppParams(a=0.3, sigma=0.01, b=0.1, eta=0.006, rho=-0.5)
        p0 = np.array([x0.a, x0.sigma, x0.b, x0.eta, x0.rho])

        def unpack(p):
            a, sigma, b, eta, rho = p
            return G2ppParams(
                a=abs(a) + 1e-4, sigma=abs(sigma), b=abs(b) + 1e-4,
                eta=abs(eta), rho=np.clip(rho, -0.999, 0.999),
            )

        def residuals(p):
            params = unpack(p)
            # Identifiability: keep a != b.
            if abs(params.a - params.b) < 1e-3:
                params.b += 1e-3
            model = G2pp(self.curve, params)
            res = []
            for pt, (mkt_price, fwd, _ann) in zip(surface, market, strict=False):
                swap = VanillaSwap(pt.expiry, pt.tenor, fwd, self.freq)
                try:
                    model_price = model.european_swaption(swap, payer=True)
                except Exception:
                    model_price = 0.0
                res.append(model_price - mkt_price)
            return np.array(res)

        sol = least_squares(residuals, p0, method="lm", max_nfev=200)
        params = unpack(sol.x)
        if abs(params.a - params.b) < 1e-3:
            params.b += 1e-3
        model = G2pp(self.curve, params)

        # Build fit table in vol terms.
        from swaptions_engine.pricing.analytic import implied_normal_vol
        table = []
        sq = 0.0
        for pt, (_, fwd, ann) in zip(surface, market, strict=False):
            swap = VanillaSwap(pt.expiry, pt.tenor, fwd, self.freq)
            mp = model.european_swaption(swap, payer=True)
            iv = implied_normal_vol(mp, fwd, fwd, pt.expiry, ann, payer=True)
            sq += (iv - pt.normal_vol) ** 2
            table.append({
                "expiry": pt.expiry, "tenor": pt.tenor,
                "market_vol_bp": pt.normal_vol * 1e4,
                "model_vol_bp": iv * 1e4,
                "diff_bp": (iv - pt.normal_vol) * 1e4,
            })
        rmse = np.sqrt(sq / len(surface)) * 1e4

        logger.info("G2++ calibrated: a=%.3f sigma=%.4f b=%.3f eta=%.4f rho=%.3f"
                    " | RMSE %.2f bp", params.a, params.sigma, params.b,
                    params.eta, params.rho, rmse)
        return CalibrationResult(params=params, rmse_vol=rmse,
                                 fit_table=table, success=sol.success)
