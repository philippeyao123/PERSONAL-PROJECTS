#!/usr/bin/env python3
"""Independent analytic and QuantLib cross-checks for qf-rates."""

import argparse
import csv
import math
from math import erf, exp, log, pi, sqrt
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


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
    expiry_years: int = 2,
    tenor_years: int = 5,
    notional: float = 1_000_000.0,
) -> Any:
    start = calendar.advance(
        today, ql.Period(expiry_years, ql.Years), ql.Unadjusted
    )
    end = calendar.advance(
        start, ql.Period(tenor_years, ql.Years), ql.Unadjusted
    )
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
        notional,
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


def make_quantlib_environment(
    ql: Any, flat_rate: float = 0.025
) -> Tuple[Any, Any, Any, Any, Any]:
    today = ql.Date(1, ql.January, 2024)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.SimpleDayCounter()
    calendar = ql.NullCalendar()
    curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, flat_rate, day_count, ql.Continuous)
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


def set_qf_parameters(qf: Any, values: Sequence[float]) -> Any:
    parameters = qf.G2ppParameters()
    (
        parameters.a,
        parameters.b,
        parameters.sigma,
        parameters.eta,
        parameters.rho,
    ) = values
    return parameters


def qf_g2pp_swaption_price(
    qf: Any,
    flat_rate: float,
    parameter_values: Sequence[float],
    expiry: int,
    tenor: int,
    strike: float,
    notional: float = 1.0,
) -> float:
    curve = qf.FlatYieldCurve(flat_rate)
    parameters = set_qf_parameters(qf, parameter_values)
    model = qf.G2ppModel(curve, parameters)
    option = qf.EuropeanSwaption(
        float(expiry),
        qf.Schedule(float(expiry), float(expiry + tenor), 1.0),
        strike,
        notional,
        qf.OptionType.Call,
    )
    return qf.g2pp_european_swaption(model, option)


def quantlib_g2pp_swaption_price(
    ql: Any,
    flat_rate: float,
    parameter_values: Sequence[float],
    expiry: int,
    tenor: int,
    strike: float,
    notional: float = 1.0,
) -> float:
    today, day_count, calendar, curve, index = make_quantlib_environment(
        ql, flat_rate
    )
    swap = make_quantlib_swap(
        ql,
        curve,
        index,
        calendar,
        day_count,
        today,
        strike,
        expiry_years=expiry,
        tenor_years=tenor,
        notional=notional,
    )
    exercise_date = calendar.advance(
        today, ql.Period(expiry, ql.Years), ql.Unadjusted
    )
    swaption = ql.Swaption(swap, ql.EuropeanExercise(exercise_date))
    a, b, sigma, eta, rho = parameter_values
    model = ql.G2(curve, a, sigma, b, eta, rho)
    swaption.setPricingEngine(ql.G2SwaptionEngine(model, 7.0, 64))
    return swaption.NPV()


def write_g2pp_validation_grids(
    ql: Any, qf: Any, output_directory: Path
) -> Tuple[List[dict[str, float]], List[dict[str, float]]]:
    scenarios: Sequence[Tuple[str, Tuple[float, float, float, float, float]]] = (
        ("low_volatility", (0.10, 0.30, 0.005, 0.0075, -0.70)),
        ("base", (0.10, 0.30, 0.010, 0.0150, -0.70)),
        ("high_volatility", (0.10, 0.30, 0.020, 0.0300, -0.70)),
        ("fast_mean_reversion", (0.30, 0.80, 0.010, 0.0150, -0.70)),
        ("weak_correlation", (0.10, 0.30, 0.010, 0.0150, -0.10)),
    )
    flat_rate = 0.025
    grid_rows: List[dict[str, float]] = []
    for scenario, parameter_values in scenarios:
        for expiry in (1, 2, 5):
            for tenor in (2, 5, 10):
                curve = qf.FlatYieldCurve(flat_rate)
                par_swap = qf.make_vanilla_swap(
                    float(expiry),
                    float(expiry + tenor),
                    flat_rate,
                    1.0,
                )
                par_rate = par_swap.par_rate(curve, curve)
                for moneyness_bp in (-100.0, 0.0, 100.0):
                    strike = par_rate + moneyness_bp * 1.0e-4
                    qf_price = qf_g2pp_swaption_price(
                        qf,
                        flat_rate,
                        parameter_values,
                        expiry,
                        tenor,
                        strike,
                    )
                    ql_price = quantlib_g2pp_swaption_price(
                        ql,
                        flat_rate,
                        parameter_values,
                        expiry,
                        tenor,
                        strike,
                    )
                    absolute_difference = abs(qf_price - ql_price)
                    grid_rows.append(
                        {
                            "scenario": scenario,
                            "expiry": float(expiry),
                            "tenor": float(tenor),
                            "moneyness_basis_points": moneyness_bp,
                            "strike": strike,
                            "qf_rates": qf_price,
                            "quantlib": ql_price,
                            "absolute_difference": absolute_difference,
                            "difference_basis_points_notional": (
                                absolute_difference * 1.0e4
                            ),
                            "relative_difference": relative_error(
                                qf_price, ql_price
                            ),
                        }
                    )

    grid_path = output_directory / "quantlib_g2pp_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(grid_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grid_rows)

    selected_cases: Sequence[
        Tuple[
            str,
            Tuple[float, float, float, float, float],
            int,
            int,
            float,
        ]
    ] = (
        ("low_vol_short", scenarios[0][1], 1, 2, 0.0),
        ("base_medium", scenarios[1][1], 2, 5, 0.0),
        ("high_vol_long", scenarios[2][1], 5, 10, 0.0),
        ("weak_corr_long", scenarios[4][1], 2, 10, 0.0),
    )
    risk_rows: List[dict[str, float]] = []
    curve_bump = 1.0e-4
    volatility_bump = 1.0e-4
    for case, parameter_values, expiry, tenor, moneyness_bp in selected_cases:
        curve = qf.FlatYieldCurve(flat_rate)
        par_swap = qf.make_vanilla_swap(
            float(expiry), float(expiry + tenor), flat_rate, 1.0
        )
        strike = (
            par_swap.par_rate(curve, curve) + moneyness_bp * 1.0e-4
        )
        qf_base = qf_g2pp_swaption_price(
            qf, flat_rate, parameter_values, expiry, tenor, strike
        )
        ql_base = quantlib_g2pp_swaption_price(
            ql, flat_rate, parameter_values, expiry, tenor, strike
        )
        qf_curve_bumped = qf_g2pp_swaption_price(
            qf,
            flat_rate - curve_bump,
            parameter_values,
            expiry,
            tenor,
            strike,
        )
        ql_curve_bumped = quantlib_g2pp_swaption_price(
            ql,
            flat_rate - curve_bump,
            parameter_values,
            expiry,
            tenor,
            strike,
        )
        a, b, sigma, eta, rho = parameter_values
        volatility_bumped = (
            a,
            b,
            sigma + volatility_bump,
            eta + volatility_bump,
            rho,
        )
        qf_vol_bumped = qf_g2pp_swaption_price(
            qf,
            flat_rate,
            volatility_bumped,
            expiry,
            tenor,
            strike,
        )
        ql_vol_bumped = quantlib_g2pp_swaption_price(
            ql,
            flat_rate,
            volatility_bumped,
            expiry,
            tenor,
            strike,
        )
        qf_curve_dv01 = qf_curve_bumped - qf_base
        ql_curve_dv01 = ql_curve_bumped - ql_base
        qf_volatility_vega = qf_vol_bumped - qf_base
        ql_volatility_vega = ql_vol_bumped - ql_base
        price_gap = abs(qf_base - ql_base)
        risk_rows.append(
            {
                "case": case,
                "expiry": float(expiry),
                "tenor": float(tenor),
                "moneyness_basis_points": moneyness_bp,
                "qf_price": qf_base,
                "quantlib_price": ql_base,
                "qf_curve_dv01": qf_curve_dv01,
                "quantlib_curve_dv01": ql_curve_dv01,
                "curve_dv01_relative_difference": relative_error(
                    qf_curve_dv01, ql_curve_dv01
                ),
                "qf_joint_volatility_vega": qf_volatility_vega,
                "quantlib_joint_volatility_vega": ql_volatility_vega,
                "volatility_vega_relative_difference": relative_error(
                    qf_volatility_vega, ql_volatility_vega
                ),
                "price_gap_in_quantlib_dv01_units": (
                    price_gap / max(abs(ql_curve_dv01), 1.0e-16)
                ),
                "price_gap_in_quantlib_volatility_bump_units": (
                    price_gap
                    / max(abs(ql_volatility_vega), 1.0e-16)
                ),
            }
        )

    risk_path = output_directory / "g2pp_risk_validation.csv"
    with risk_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(risk_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(risk_rows)
    return grid_rows, risk_rows


def cross_check_quantlib(
    ql: Any, qf: Any, output_directory: Optional[Path] = None
) -> None:
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
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        black_reference = black76(100.0, 100.0, 0.20, 1.0, 0.95)
        black_quantlib = ql.blackFormula(
            ql.Option.Call, 100.0, 100.0, 0.20, 0.95
        )
        normal_reference = bachelier(0.03, 0.03, 0.01, 1.0, 0.97)
        normal_quantlib = ql.bachelierBlackFormula(
            ql.Option.Call, 0.03, 0.03, 0.01, 0.97
        )
        validation_rows = (
            ("Black-76 ATM", black_reference, black_quantlib),
            ("Bachelier ATM", normal_reference, normal_quantlib),
            ("2Yx5Y swap par rate", cpp_par, ql_par),
            ("2Yx5Y swap NPV", cpp_npv, ql_npv),
            ("G2++ European swaption", cpp_price, ql_price),
        )
        with (output_directory / "quantlib_validation.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "quantity",
                    "qf_rates",
                    "quantlib",
                    "absolute_difference",
                    "relative_difference",
                )
            )
            for quantity, actual, reference in validation_rows:
                writer.writerow(
                    (
                        quantity,
                        f"{actual:.12f}",
                        f"{reference:.12f}",
                        f"{abs(actual - reference):.12f}",
                        f"{relative_error(actual, reference):.12f}",
                    )
                )
        ql_parameters = list(ql_calibrated_model.params())
        with (output_directory / "calibration_validation.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "implementation",
                    "rmse_normal_volatility",
                    "a",
                    "b",
                    "sigma",
                    "eta",
                    "rho",
                )
            )
            writer.writerow(
                (
                    "qf-rates",
                    f"{cpp_calibration.rmse:.10f}",
                    f"{cpp_params.a:.8f}",
                    f"{cpp_params.b:.8f}",
                    f"{cpp_params.sigma:.8f}",
                    f"{cpp_params.eta:.8f}",
                    f"{cpp_params.rho:.8f}",
                )
            )
            writer.writerow(
                (
                    "QuantLib 1.43",
                    f"{ql_rmse:.10f}",
                    *(f"{value:.8f}" for value in ql_parameters),
                )
            )
        grid_rows, risk_rows = write_g2pp_validation_grids(
            ql, qf, output_directory
        )
        material_rows = [
            row for row in grid_rows if row["quantlib"] >= 1.0e-4
        ]
        assert len(grid_rows) == 135
        assert len(risk_rows) == 4
        assert material_rows
        assert all(
            math.isfinite(value)
            for row in grid_rows
            for value in (
                row["qf_rates"],
                row["quantlib"],
                row["relative_difference"],
            )
        )
        ordered_material_errors = sorted(
            row["relative_difference"] for row in material_rows
        )
        p95_index = math.ceil(0.95 * len(ordered_material_errors)) - 1
        p95_error = ordered_material_errors[p95_index]
        max_absolute_difference = max(
            row["difference_basis_points_notional"] for row in grid_rows
        )
        max_dv01_difference = max(
            row["curve_dv01_relative_difference"] for row in risk_rows
        )
        max_volatility_difference = max(
            row["volatility_vega_relative_difference"] for row in risk_rows
        )
        # Broad regression gates detect numerical drift without claiming exact
        # equivalence between engines that use different integration schemes.
        assert p95_error < 0.06
        assert max_absolute_difference < 5.0
        assert max_dv01_difference < 0.06
        assert max_volatility_difference < 0.02
        print(f"quantlib_g2_grid_cells={len(grid_rows)}")
        print(
            "quantlib_g2_grid_material_max_relative_error="
            f"{max(row['relative_difference'] for row in material_rows):.8f}"
        )
        print(f"quantlib_g2_grid_material_p95_relative_error={p95_error:.8f}")
        print(f"quantlib_g2_grid_max_absolute_difference_bp={max_absolute_difference:.8f}")
        print(f"quantlib_g2_risk_max_dv01_relative_error={max_dv01_difference:.8f}")
        print(
            "quantlib_g2_risk_max_volatility_relative_error="
            f"{max_volatility_difference:.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-quantlib", action="store_true")
    parser.add_argument("--require-bindings", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Write QuantLib and calibration CSV files to this directory",
    )
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
        cross_check_quantlib(ql, qf, args.output_directory)
    elif qf is not None:
        cpp_black = qf.black76(
            qf.OptionType.Call, 100.0, 100.0, 0.20, 1.0, 0.95
        ).price
        assert abs(cpp_black - values["black76_atm"]) < 1.0e-12
        print(f"bindings_black76_atm={cpp_black:.12f}")


if __name__ == "__main__":
    main()
