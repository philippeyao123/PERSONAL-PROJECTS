#!/usr/bin/env python3
"""Exercise every major pricing family exposed by the optional pybind11 module."""

import math

import qf_rates_python as qf


def main() -> None:
    curve = qf.FlatYieldCurve(0.025)
    swap = qf.make_vanilla_swap(0.0, 7.0, 0.03, 1_000_000.0)
    par_rate = swap.par_rate(curve, curve)
    assert math.isfinite(swap.npv(curve, curve))
    assert par_rate > 0.0

    parameters = qf.G2ppParameters()
    model = qf.G2ppModel(curve, parameters)
    european = qf.EuropeanSwaption(
        2.0, qf.Schedule(2.0, 7.0, 1.0), par_rate, 1_000_000.0, qf.OptionType.Call
    )
    deterministic = qf.g2pp_european_swaption(model, european)
    mc_config = qf.MonteCarloConfig()
    mc_config.paths = 5_000
    mc_config.time_steps = 48
    mc = qf.g2pp_european_swaption_mc(model, european, mc_config)
    time_config = qf.MonteCarloTimeConvergenceConfig()
    time_config.paths = 2_000
    time_config.finest_time_steps = 48
    time_config.time_steps = [12, 24, 48]
    time_convergence = qf.g2pp_european_swaption_mc_time_convergence(
        model, european, time_config
    )
    assert deterministic > 0.0
    assert mc.confidence_low <= deterministic + 5.0 * mc.standard_error
    assert time_convergence[-1].paired_bias_vs_finest == 0.0

    bermudan = qf.BermudanSwaption()
    bermudan.exercise_times = [1.0, 2.0, 3.0]
    bermudan.maturity = 7.0
    bermudan.strike = par_rate
    bermudan.notional = 1_000_000.0
    lsm_config = qf.LsmConfig()
    lsm_config.paths = 2_000
    lsm = qf.g2pp_bermudan_lsm(model, bermudan, lsm_config)
    oos_config = qf.LsmOutOfSampleConfig()
    oos_config.training_paths = 1_000
    oos_config.valuation_paths = 2_000
    oos = qf.g2pp_bermudan_lsm_out_of_sample(model, bermudan, oos_config)
    assert lsm.price > 0.0
    assert oos.price > 0.0

    quote = qf.SwaptionQuote()
    quote.expiry = 1.0
    quote.tenor = 5.0
    quote.strike = 0.026
    quote.normal_volatility = 0.006
    calibration = qf.calibrate_g2pp(curve, [quote])
    multi_config = qf.G2ppMultiStartConfig()
    multi_config.starts = 2
    multistart = qf.calibrate_g2pp_multistart(curve, [quote], multi_config)
    assert math.isfinite(calibration.rmse)
    assert len(calibration.diagnostics) == 1
    assert multistart.best.rmse <= multistart.runs[0].calibration.rmse

    interpolated = qf.InterpolatedYieldCurve(
        [0.0, 1.0, 2.0, 5.0, 10.0],
        [1.0, 0.98, 0.955, 0.88, 0.75],
    )
    dv01 = qf.swap_dv01(swap, interpolated)
    assert math.isfinite(dv01.parallel_dv01)
    scenarios = qf.run_bachelier_volatility_scenarios(
        qf.OptionType.Call, 0.03, 0.03, 0.006, 2.0, 4.5
    )
    assert scenarios[0].change > 0.0
    assert scenarios[1].change < 0.0

    print(
        "bindings_smoke=ok "
        f"swap_par={par_rate:.10f} european={deterministic:.6f} "
        f"mc={mc.price:.6f} bermudan={lsm.price:.6f} oos={oos.price:.6f} "
        f"rmse={calibration.rmse:.10f} multistart={multistart.best.rmse:.10f}"
    )


if __name__ == "__main__":
    main()
