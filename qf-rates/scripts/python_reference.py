#!/usr/bin/env python3
"""Independent analytic and QuantLib cross-checks for qf-rates."""

import argparse
import math
from math import erf, exp, log, pi, sqrt
from typing import Any, List, Sequence, Tuple


def cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black76(
    forward: float, strike: float, vol: float, expiry: float, discount: float
) -> float:
    std = vol * sqrt(expiry)
    d1 = (log(forward / strike) + 0.5 * std * std) / std
    return discount * (forward * cdf(d1) - strike * cdf(d1 - std))


def bachelier(
    forward: float, strike: float, vol: float, expiry: float, discount: float
) -> float:
    std = vol * sqrt(expiry)
    d = (forward - strike) / std
    pdf = exp(-0.5 * d * d) / sqrt(2.0 * pi)
    return discount * ((forward - strike) * cdf(d) + std * pdf)


def relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1.0e-16)


def load_optional_modules(require_quantlib: bool, require_bindings: bool) -> Tuple[Any, Any]:
    ql = None
    qf = None
    try:
        import QuantLib as quantlib  # type: ignore

        ql = quantlib
    except ImportError:
        if require_quantlib:
            raise RuntimeError("QuantLib is required but could not be imported")

    try:
        import qf_rates_python as bindings  # type: ignore

        qf = bindings
    except ImportError:
        if require_bindings:
            raise RuntimeError("qf_rates_python is required but could not be imported")
    return ql, qf


def make_quantlib_swap(
    ql: Any,
    curve: Any,
    index: Any,
    calendar: Any,
    day_count: Any,
    today: Any,
    fixed_rate: float,
) -> Any:
    start = calendar.advance(today, ql.Period(2, ql.Years), ql.Unadjusted)
    end = calendar.advance(today, ql.Period(7, ql.Years), ql.Unadjusted)
    fixed_schedule = ql.Schedule(
        start,
        end,
        ql.Period(1, ql.Years),
        calendar,
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Forward,
        False,
    )
    floating_schedule = ql.Schedule(
        start,
        end,
        ql.Period(6, ql.Months),
        calendar,
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Forward,
        False,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        1_000_000.0,
        fixed_schedule,
        fixed_rate,
        day_count,
        floating_schedule,
        index,
        0.0,
        day_count,
    )
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve))
    return swap


def make_quantlib_environment(ql: Any) -> Tuple[Any, Any, Any, Any, Any]:
    today = ql.Date(1, ql.January, 2024)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.SimpleDayCounter()
    calendar = ql.NullCalendar()
    curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.025, day_count, ql.Continuous)
    )
    index = ql.IborIndex(
        "Flat6M",
        ql.Period(6, ql.Months),
        0,
        ql.USDCurrency(),
        calendar,
        ql.Unadjusted,
        False,
        day_count,
        curve,
    )
    return today, day_count, calendar, curve, index


def cross_check_quantlib(ql: Any, qf: Any) -> None:
    today, day_count, calendar, ql_curve, index = make_quantlib_environment(ql)
    cpp_curve = qf.FlatYieldCurve(0.025)

    cpp_swap = qf.make_vanilla_swap(2.0, 7.0, 0.03, 1_000_000.0)
    ql_swap = make_quantlib_swap(
        ql, ql_curve, index, calendar, day_count, today, 0.03
    )
    cpp_par = cpp_swap.par_rate(cpp_curve, cpp_curve)
    cpp_npv = cpp_swap.npv(cpp_curve, cpp_curve)
    ql_par = ql_swap.fairRate()
    ql_npv = ql_swap.NPV()
    assert abs(cpp_par - ql_par) < 1.0e-12
    assert abs(cpp_npv - ql_npv) < 1.0e-7
    print(f"quantlib_swap_par_qf={cpp_par:.12f}")
    print(f"quantlib_swap_par_ql={ql_par:.12f}")
    print(f"quantlib_swap_npv_qf={cpp_npv:.8f}")
    print(f"quantlib_swap_npv_ql={ql_npv:.8f}")

    parameters = qf.G2ppParameters()
    cpp_model = qf.G2ppModel(cpp_curve, parameters)
    cpp_option = qf.EuropeanSwaption(
        2.0,
        qf.Schedule(2.0, 7.0, 1.0),
        cpp_par,
        1_000_000.0,
        qf.OptionType.Call,
    )
    cpp_price = qf.g2pp_european_swaption(cpp_model, cpp_option)

    ql_atm_swap = make_quantlib_swap(
        ql, ql_curve, index, calendar, day_count, today, cpp_par
    )
    expiry_date = calendar.advance(
        today, ql.Period(2, ql.Years), ql.Unadjusted
    )
    ql_swaption = ql.Swaption(ql_atm_swap, ql.EuropeanExercise(expiry_date))
    ql_model = ql.G2(
        ql_curve,
        parameters.a,
        parameters.sigma,
        parameters.b,
        parameters.eta,
        parameters.rho,
    )
    ql_swaption.setPricingEngine(ql.G2SwaptionEngine(ql_model, 7.0, 64))
    ql_price = ql_swaption.NPV()
    g2_relative_error = relative_error(cpp_price, ql_price)
    assert g2_relative_error < 0.01
    print(f"quantlib_g2_swaption_qf={cpp_price:.8f}")
    print(f"quantlib_g2_swaption_ql={ql_price:.8f}")
    print(f"quantlib_g2_relative_error={g2_relative_error:.8f}")

    quote_data: Sequence[Tuple[int, int, float, float]] = (
        (1, 5, 0.0060, 0.026),
        (2, 5, 0.0065, 0.026),
        (3, 5, 0.0068, 0.026),
        (1, 10, 0.0064, 0.026),
        (2, 10, 0.0069, 0.026),
        (3, 10, 0.0072, 0.026),
    )
    cpp_quotes: List[Any] = []
    helpers: List[Any] = []
    for expiry, tenor, vol, strike in quote_data:
        quote = qf.SwaptionQuote()
        quote.expiry = float(expiry)
        quote.tenor = float(tenor)
        quote.normal_volatility = vol
        quote.strike = strike
        cpp_quotes.append(quote)

        helper = ql.SwaptionHelper(
            ql.Period(expiry, ql.Years),
            ql.Period(tenor, ql.Years),
            ql.QuoteHandle(ql.SimpleQuote(vol)),
            index,
            ql.Period(1, ql.Years),
            day_count,
            day_count,
            ql_curve,
            ql.BlackCalibrationHelper.ImpliedVolError,
            strike,
            1.0,
            ql.Normal,
        )
        helpers.append(helper)

    cpp_calibration = qf.calibrate_g2pp(cpp_curve, cpp_quotes)
    assert math.isfinite(cpp_calibration.rmse)
    assert cpp_calibration.rmse < 0.001

    ql_calibrated_model = ql.G2(ql_curve)
    ql_engine = ql.G2SwaptionEngine(ql_calibrated_model, 7.0, 64)
    for helper in helpers:
        helper.setPricingEngine(ql_engine)
    ql_calibrated_model.calibrate(
        helpers,
        ql.LevenbergMarquardt(),
        ql.EndCriteria(1000, 100, 1.0e-8, 1.0e-8, 1.0e-8),
    )
    ql_errors = [helper.calibrationError() for helper in helpers]
    ql_rmse = sqrt(sum(error * error for error in ql_errors) / len(ql_errors))
    assert math.isfinite(ql_rmse)
    assert ql_rmse < 0.001
    cpp_params = cpp_calibration.parameters
    print(f"quantlib_calibration_qf_rmse={cpp_calibration.rmse:.10f}")
    print(f"quantlib_calibration_ql_rmse={ql_rmse:.10f}")
    print(
        "quantlib_calibration_qf_parameters="
        f"{cpp_params.a:.8f},{cpp_params.b:.8f},{cpp_params.sigma:.8f},"
        f"{cpp_params.eta:.8f},{cpp_params.rho:.8f}"
    )
    print(
        "quantlib_calibration_ql_parameters="
        + ",".join(f"{value:.8f}" for value in ql_calibrated_model.params())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-quantlib", action="store_true")
    parser.add_argument("--require-bindings", action="store_true")
    args = parser.parse_args()

    values = {
        "flat_curve_discount_5y": exp(-0.03 * 5.0),
        "black76_atm": black76(100.0, 100.0, 0.20, 1.0, 0.95),
        "bachelier_atm": bachelier(0.03, 0.03, 0.01, 1.0, 0.97),
    }
    for name, value in values.items():
        print(f"{name}={value:.12f}")

    ql, qf = load_optional_modules(args.require_quantlib, args.require_bindings)
    if ql is not None:
        black = ql.blackFormula(ql.Option.Call, 100.0, 100.0, 0.20, 0.95)
        normal = ql.bachelierBlackFormula(
            ql.Option.Call, 0.03, 0.03, 0.01, 0.97
        )
        assert abs(black - values["black76_atm"]) < 1.0e-12
        assert abs(normal - values["bachelier_atm"]) < 1.0e-12
        print(f"quantlib_black76_atm={black:.12f}")
        print(f"quantlib_bachelier_atm={normal:.12f}")
    else:
        print("QuantLib not installed: standard-library references completed")

    if ql is not None and qf is not None:
        cross_check_quantlib(ql, qf)
    elif qf is not None:
        cpp_black = qf.black76(
            qf.OptionType.Call, 100.0, 100.0, 0.20, 1.0, 0.95
        ).price
        assert abs(cpp_black - values["black76_atm"]) < 1.0e-12
        print(f"bindings_black76_atm={cpp_black:.12f}")


if __name__ == "__main__":
    main()
