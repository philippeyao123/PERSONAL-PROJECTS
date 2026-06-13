"""Test suite. Priorities: arbitrage-free curve, G2++ analytic == MC,
LSM single-exercise == European and Bermudan >= European, calibration fit,
and risk-sign sanity.
"""
from __future__ import annotations

import numpy as np
import pytest

from swaptions_engine.calibration.calibrate import (
    G2ppCalibrator,
    SurfacePoint,
)
from swaptions_engine.curve.yield_curve import Curve
from swaptions_engine.instruments.swap import Swaption, VanillaSwap
from swaptions_engine.models.g2pp import G2pp, G2ppParams
from swaptions_engine.models.sabr import SABRParams, sabr_normal_vol
from swaptions_engine.pricing.analytic import (
    bachelier_swaption,
    black_swaption,
    implied_normal_vol,
)
from swaptions_engine.pricing.bermudan import BermudanLSM
from swaptions_engine.pricing.montecarlo import G2ppSimulator
from swaptions_engine.risk.engine import SwaptionRisk


def _g2_bond(m, t, T, x, y):
    return m._bond_coef_A(t, T) * np.exp(-m._Bx(t, T) * x - m._By(t, T) * y)


# ----------------------------- curve -----------------------------
def test_curve_discount_monotone():
    c = Curve.flat(0.03)
    assert c.discount(1) > c.discount(2) > c.discount(5)
    assert c.discount(0.0) == pytest.approx(1.0, abs=1e-6)


def test_curve_forward_rate_flat():
    c = Curve.flat(0.04)
    # Flat continuously-compounded 4% => simple fwd slightly above 4%.
    f = c.forward_rate(1, 2)
    assert 0.039 < f < 0.042


def test_curve_bump_pillar():
    c = Curve.flat(0.03)
    c2 = c.bump_pillar(0, 1e-4)
    assert c2.discount(c.t[0]) < c.discount(c.t[0])


# ----------------------------- analytic -----------------------------
def test_bachelier_intrinsic_at_zero_vol():
    p = bachelier_swaption(0.03, 0.02, 1.0, 0.0, 4.0, payer=True)
    assert p == pytest.approx(4.0 * 0.01)


def test_black_put_call_parity():
    f, k, t, v, ann = 0.03, 0.03, 2.0, 0.20, 4.0
    call = black_swaption(f, k, t, v, ann, payer=True)
    put = black_swaption(f, k, t, v, ann, payer=False)
    # payer - receiver = ann*(f - k) = 0 at ATM
    assert call - put == pytest.approx(0.0, abs=1e-10)


def test_implied_normal_vol_roundtrip():
    f, k, t, ann = 0.025, 0.025, 3.0, 4.5
    vol = 0.007
    price = bachelier_swaption(f, k, t, vol, ann, payer=True)
    iv = implied_normal_vol(price, f, k, t, ann, payer=True)
    assert iv == pytest.approx(vol, abs=1e-5)


# ----------------------------- SABR -----------------------------
def test_sabr_atm_matches_alpha_beta0():
    p = SABRParams(alpha=0.006, beta=0.0, rho=-0.3, nu=0.4)
    v = sabr_normal_vol(0.02, 0.02, 1.0, p)
    # ATM beta=0 normal vol ~ alpha * (1 + small correction)
    assert abs(v - 0.006) / 0.006 < 0.05


def test_sabr_smile_is_smile():
    p = SABRParams(alpha=0.006, beta=0.0, rho=-0.2, nu=0.5)
    f = 0.02
    lo = sabr_normal_vol(f, f - 0.01, 2.0, p)
    atm = sabr_normal_vol(f, f, 2.0, p)
    hi = sabr_normal_vol(f, f + 0.01, 2.0, p)
    assert lo > 0 and hi > 0 and atm > 0


# ----------------------------- G2++ analytic vs MC -----------------------------
@pytest.mark.parametrize("expiry,tenor", [(1.0, 5.0), (2.0, 5.0)])
def test_g2pp_analytic_matches_mc(expiry, tenor):
    curve = Curve.flat(0.03)
    params = G2ppParams(a=0.5, sigma=0.012, b=0.15, eta=0.008, rho=-0.6)
    model = G2pp(curve, params)
    sw0 = VanillaSwap(expiry, tenor, 0.0)
    fwd = sw0.forward_swap_rate(curve)
    swap = VanillaSwap(expiry, tenor, fwd)

    analytic = model.european_swaption(swap, payer=True)

    sim = G2ppSimulator(model, seed=1)
    times = np.linspace(0, expiry, max(13, int(expiry * 12)))
    x, y, num = sim.simulate(times, 120_000)
    xT, yT = x[:, -1], y[:, -1]
    ann = np.zeros(len(xT))
    for ti in swap.pay_times:
        ann += swap.accrual * _g2_bond(model, expiry, ti, xT, yT)
    npv = (1.0 - _g2_bond(model, expiry, swap.pay_times[-1], xT, yT)) - fwd * ann
    mc = np.mean(np.maximum(npv, 0) / num[:, -1])
    # Within 2% (MC noise + discretization).
    assert abs(analytic - mc) / mc < 0.02


# ----------------------------- Bermudan LSM -----------------------------
def test_lsm_single_exercise_equals_european():
    curve = Curve.flat(0.03)
    params = G2ppParams(a=0.5, sigma=0.012, b=0.15, eta=0.008, rho=-0.6)
    model = G2pp(curve, params)
    sw0 = VanillaSwap(1.0, 5.0, 0.0)
    fwd = sw0.forward_swap_rate(curve)
    swap = VanillaSwap(1.0, 5.0, fwd)
    euro = model.european_swaption(swap, payer=True)
    lsm = BermudanLSM(model)
    res = lsm.price(Swaption(swap, [1.0], "european"), n_paths=40_000, seed=3)
    assert abs(res["price"] - euro) < 4 * res["std_error"] + 1e-4


def test_bermudan_exceeds_european():
    curve = Curve.flat(0.03)
    params = G2ppParams(a=0.5, sigma=0.012, b=0.15, eta=0.008, rho=-0.6)
    model = G2pp(curve, params)
    sw0 = VanillaSwap(1.0, 5.0, 0.0)
    fwd = sw0.forward_swap_rate(curve)
    swap = VanillaSwap(1.0, 5.0, fwd)
    euro = model.european_swaption(swap, payer=True)
    lsm = BermudanLSM(model)
    berm = lsm.price(Swaption(swap, [1.0, 2.0, 3.0, 4.0, 5.0], "bermudan"),
                     n_paths=40_000, seed=4)
    assert berm["price"] >= euro - 4 * berm["std_error"]


# ----------------------------- calibration -----------------------------
def test_calibration_fits_surface():
    curve = Curve.flat(0.025)
    surf = [
        SurfacePoint(1, 5, 0.0075), SurfacePoint(2, 5, 0.0080),
        SurfacePoint(5, 5, 0.0078), SurfacePoint(1, 10, 0.0072),
        SurfacePoint(5, 10, 0.0074),
    ]
    res = G2ppCalibrator(curve).calibrate(surf)
    assert res.success
    assert res.rmse_vol < 5.0  # sub-5bp fit


# ----------------------------- risk -----------------------------
def test_payer_swaption_risk_signs():
    curve = Curve.flat(0.025)
    params = G2ppParams(a=0.5, sigma=0.012, b=0.15, eta=0.008, rho=-0.6)
    sw0 = VanillaSwap(2.0, 5.0, 0.0)
    fwd = sw0.forward_swap_rate(curve)
    swap = VanillaSwap(2.0, 5.0, fwd)
    risk = SwaptionRisk(curve, params)
    rep = risk.compute(lambda m: m.european_swaption(swap, True))
    assert rep.pv > 0
    assert rep.total_dv01 > 0          # payer gains as rates rise
    assert rep.vega > 0                # long optionality
    # Positive convexity: up-shock gain exceeds down-shock loss magnitude.
    assert rep.scenarios["parallel_+50bp"] > -rep.scenarios["parallel_-50bp"]


# ----------------------------- plots -----------------------------
def test_plots_generate(tmp_path):
    from swaptions_engine.plots import (
        plot_bermudan_premium,
        plot_bucketed_dv01,
        plot_calibration_fit,
    )
    fit = [{"expiry": 1, "tenor": 5, "market_vol_bp": 75, "model_vol_bp": 75.6,
            "diff_bp": 0.6}]
    assert plot_calibration_fit(fit, 1.46, tmp_path / "c.png").exists()
    assert plot_bermudan_premium([1, 2, 3], [0.01, 0.012, 0.014], 0.01,
                                 tmp_path / "b.png").exists()
    assert plot_bucketed_dv01({2.0: -95.0, 7.0: 302.0},
                              tmp_path / "d.png").exists()
