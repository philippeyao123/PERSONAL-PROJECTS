#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <utility>
#include <vector>

#include "qf/rates/calibration.hpp"
#include "qf/rates/g2pp.hpp"
#include "qf/rates/lsm.hpp"
#include "qf/rates/monte_carlo.hpp"
#include "qf/rates/options.hpp"
#include "qf/rates/risk.hpp"
#include "qf/rates/swap.hpp"
#include "qf/rates/yield_curve.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qf_rates_python, module) {
  module.doc() = "Python bindings for the qf-rates C++ pricing and risk library";

  py::enum_<qf::OptionType>(module, "OptionType")
      .value("Call", qf::OptionType::Call)
      .value("Put", qf::OptionType::Put);
  py::enum_<qf::PayReceive>(module, "PayReceive")
      .value("Pay", qf::PayReceive::Pay)
      .value("Receive", qf::PayReceive::Receive);
  py::enum_<qf::Interpolation>(module, "Interpolation")
      .value("LinearDiscount", qf::Interpolation::LinearDiscount)
      .value("LogLinearDiscount", qf::Interpolation::LogLinearDiscount);
  py::enum_<qf::LsmBasis>(module, "LsmBasis")
      .value("Linear", qf::LsmBasis::Linear)
      .value("Quadratic", qf::LsmBasis::Quadratic);

  py::class_<qf::CouponPeriod>(module, "CouponPeriod")
      .def_readonly("start", &qf::CouponPeriod::start)
      .def_readonly("end", &qf::CouponPeriod::end)
      .def_readonly("payment", &qf::CouponPeriod::payment)
      .def_readonly("accrual", &qf::CouponPeriod::accrual);
  py::class_<qf::Schedule>(module, "Schedule")
      .def(py::init<qf::Time, qf::Time, double>())
      .def_property_readonly("start", &qf::Schedule::start)
      .def_property_readonly("end", &qf::Schedule::end)
      .def_property_readonly("periods", &qf::Schedule::periods,
                             py::return_value_policy::reference_internal);

  py::class_<qf::YieldCurve, std::shared_ptr<qf::YieldCurve>>(module, "YieldCurve")
      .def("discount", &qf::YieldCurve::discount)
      .def("zero_rate", &qf::YieldCurve::zero_rate)
      .def("forward_rate", &qf::YieldCurve::forward_rate);
  py::class_<qf::FlatYieldCurve, qf::YieldCurve, std::shared_ptr<qf::FlatYieldCurve>>(
      module, "FlatYieldCurve")
      .def(py::init<qf::Rate>())
      .def_property_readonly("rate", &qf::FlatYieldCurve::rate);
  py::class_<qf::InterpolatedYieldCurve, qf::YieldCurve,
             std::shared_ptr<qf::InterpolatedYieldCurve>>(module, "InterpolatedYieldCurve")
      .def(py::init<std::vector<qf::Time>, std::vector<qf::DiscountFactor>, qf::Interpolation>(),
           py::arg("maturities"), py::arg("discounts"),
           py::arg("interpolation") = qf::Interpolation::LogLinearDiscount)
      .def_property_readonly("maturities", &qf::InterpolatedYieldCurve::maturities)
      .def_property_readonly("discounts", &qf::InterpolatedYieldCurve::discounts);

  py::class_<qf::InterestRateSwap>(module, "InterestRateSwap")
      .def("npv", [](const qf::InterestRateSwap& swap, const qf::YieldCurve& discount,
                     const qf::YieldCurve& forward) { return swap.npv(discount, forward); })
      .def("par_rate",
           [](const qf::InterestRateSwap& swap, const qf::YieldCurve& discount,
              const qf::YieldCurve& forward) { return swap.par_rate(discount, forward); })
      .def("annuity", &qf::InterestRateSwap::annuity);
  module.def("make_vanilla_swap", &qf::make_vanilla_swap, py::arg("start"), py::arg("maturity"),
             py::arg("fixed_rate"), py::arg("notional") = 1.0,
             py::arg("fixed_direction") = qf::PayReceive::Pay, py::arg("fixed_frequency") = 1.0,
             py::arg("floating_frequency") = 0.5, py::arg("spread") = 0.0);

  py::class_<qf::OptionResult>(module, "OptionResult")
      .def_readonly("price", &qf::OptionResult::price)
      .def_readonly("delta", &qf::OptionResult::delta)
      .def_readonly("gamma", &qf::OptionResult::gamma)
      .def_readonly("vega", &qf::OptionResult::vega);
  module.def("black76", &qf::black76, py::arg("type"), py::arg("forward"), py::arg("strike"),
             py::arg("volatility"), py::arg("expiry"), py::arg("discount") = 1.0,
             py::arg("notional") = 1.0);
  module.def("bachelier", &qf::bachelier, py::arg("type"), py::arg("forward"), py::arg("strike"),
             py::arg("normal_volatility"), py::arg("expiry"), py::arg("discount") = 1.0,
             py::arg("notional") = 1.0);

  py::class_<qf::G2ppParameters>(module, "G2ppParameters")
      .def(py::init<>())
      .def_readwrite("a", &qf::G2ppParameters::a)
      .def_readwrite("b", &qf::G2ppParameters::b)
      .def_readwrite("sigma", &qf::G2ppParameters::sigma)
      .def_readwrite("eta", &qf::G2ppParameters::eta)
      .def_readwrite("rho", &qf::G2ppParameters::rho);
  py::class_<qf::FactorState>(module, "FactorState")
      .def(py::init<>())
      .def_readwrite("x", &qf::FactorState::x)
      .def_readwrite("y", &qf::FactorState::y);
  py::class_<qf::EuropeanSwaption>(module, "EuropeanSwaption")
      .def(py::init<qf::Time, qf::Schedule, qf::Rate, qf::Money, qf::OptionType>(),
           py::arg("expiry"), py::arg("underlying_schedule"), py::arg("strike"),
           py::arg("notional") = 1.0, py::arg("type") = qf::OptionType::Call)
      .def_readwrite("expiry", &qf::EuropeanSwaption::expiry)
      .def_readwrite("strike", &qf::EuropeanSwaption::strike)
      .def_readwrite("notional", &qf::EuropeanSwaption::notional)
      .def_readwrite("type", &qf::EuropeanSwaption::type);
  py::class_<qf::G2ppModel>(module, "G2ppModel")
      .def(py::init([](std::shared_ptr<qf::YieldCurve> curve, qf::G2ppParameters parameters) {
             return qf::G2ppModel(std::move(curve), parameters);
           }),
           py::arg("curve"), py::arg("parameters") = qf::G2ppParameters{})
      .def("discount_bond", &qf::G2ppModel::discount_bond, py::arg("observation"),
           py::arg("maturity"), py::arg("state") = qf::FactorState{})
      .def("integrated_variance", &qf::G2ppModel::integrated_variance)
      .def_property_readonly("parameters", &qf::G2ppModel::parameters,
                             py::return_value_policy::reference_internal);
  module.def("g2pp_european_swaption", &qf::g2pp_european_swaption, py::arg("model"),
             py::arg("swaption"), py::arg("quadrature_order") = 8U);

  py::class_<qf::MonteCarloConfig>(module, "MonteCarloConfig")
      .def(py::init<>())
      .def_readwrite("paths", &qf::MonteCarloConfig::paths)
      .def_readwrite("time_steps", &qf::MonteCarloConfig::time_steps)
      .def_readwrite("seed", &qf::MonteCarloConfig::seed)
      .def_readwrite("antithetic", &qf::MonteCarloConfig::antithetic);
  py::class_<qf::MonteCarloResult>(module, "MonteCarloResult")
      .def_readonly("price", &qf::MonteCarloResult::price)
      .def_readonly("standard_error", &qf::MonteCarloResult::standard_error)
      .def_readonly("confidence_low", &qf::MonteCarloResult::confidence_low)
      .def_readonly("confidence_high", &qf::MonteCarloResult::confidence_high)
      .def_readonly("paths", &qf::MonteCarloResult::paths);
  module.def("g2pp_european_swaption_mc", &qf::g2pp_european_swaption_mc, py::arg("model"),
             py::arg("swaption"), py::arg("config") = qf::MonteCarloConfig{});
  py::class_<qf::MonteCarloTimeConvergenceConfig>(module, "MonteCarloTimeConvergenceConfig")
      .def(py::init<>())
      .def_readwrite("paths", &qf::MonteCarloTimeConvergenceConfig::paths)
      .def_readwrite("finest_time_steps", &qf::MonteCarloTimeConvergenceConfig::finest_time_steps)
      .def_readwrite("time_steps", &qf::MonteCarloTimeConvergenceConfig::time_steps)
      .def_readwrite("seed", &qf::MonteCarloTimeConvergenceConfig::seed)
      .def_readwrite("antithetic", &qf::MonteCarloTimeConvergenceConfig::antithetic);
  py::class_<qf::MonteCarloTimeConvergenceResult>(module, "MonteCarloTimeConvergenceResult")
      .def_readonly("time_steps", &qf::MonteCarloTimeConvergenceResult::time_steps)
      .def_readonly("price", &qf::MonteCarloTimeConvergenceResult::price)
      .def_readonly("standard_error", &qf::MonteCarloTimeConvergenceResult::standard_error)
      .def_readonly("paired_bias_vs_finest",
                    &qf::MonteCarloTimeConvergenceResult::paired_bias_vs_finest)
      .def_readonly("paired_bias_standard_error",
                    &qf::MonteCarloTimeConvergenceResult::paired_bias_standard_error);
  module.def("g2pp_european_swaption_mc_time_convergence",
             &qf::g2pp_european_swaption_mc_time_convergence, py::arg("model"), py::arg("swaption"),
             py::arg("config") = qf::MonteCarloTimeConvergenceConfig{});

  py::class_<qf::BermudanSwaption>(module, "BermudanSwaption")
      .def(py::init<>())
      .def_readwrite("exercise_times", &qf::BermudanSwaption::exercise_times)
      .def_readwrite("maturity", &qf::BermudanSwaption::maturity)
      .def_readwrite("strike", &qf::BermudanSwaption::strike)
      .def_readwrite("notional", &qf::BermudanSwaption::notional)
      .def_readwrite("type", &qf::BermudanSwaption::type)
      .def_readwrite("fixed_frequency", &qf::BermudanSwaption::fixed_frequency);
  py::class_<qf::LsmConfig>(module, "LsmConfig")
      .def(py::init<>())
      .def_readwrite("paths", &qf::LsmConfig::paths)
      .def_readwrite("steps_per_year", &qf::LsmConfig::steps_per_year)
      .def_readwrite("seed", &qf::LsmConfig::seed)
      .def_readwrite("basis", &qf::LsmConfig::basis);
  py::class_<qf::LsmResult>(module, "LsmResult")
      .def_readonly("price", &qf::LsmResult::price)
      .def_readonly("standard_error", &qf::LsmResult::standard_error)
      .def_readonly("exercise_probabilities", &qf::LsmResult::exercise_probabilities);
  module.def("g2pp_bermudan_lsm", &qf::g2pp_bermudan_lsm, py::arg("model"), py::arg("swaption"),
             py::arg("config") = qf::LsmConfig{});
  py::class_<qf::LsmOutOfSampleConfig>(module, "LsmOutOfSampleConfig")
      .def(py::init<>())
      .def_readwrite("training_paths", &qf::LsmOutOfSampleConfig::training_paths)
      .def_readwrite("valuation_paths", &qf::LsmOutOfSampleConfig::valuation_paths)
      .def_readwrite("steps_per_year", &qf::LsmOutOfSampleConfig::steps_per_year)
      .def_readwrite("training_seed", &qf::LsmOutOfSampleConfig::training_seed)
      .def_readwrite("valuation_seed", &qf::LsmOutOfSampleConfig::valuation_seed)
      .def_readwrite("basis", &qf::LsmOutOfSampleConfig::basis);
  py::class_<qf::LsmOutOfSampleResult>(module, "LsmOutOfSampleResult")
      .def_readonly("training_price", &qf::LsmOutOfSampleResult::training_price)
      .def_readonly("price", &qf::LsmOutOfSampleResult::price)
      .def_readonly("standard_error", &qf::LsmOutOfSampleResult::standard_error)
      .def_readonly("exercise_probabilities", &qf::LsmOutOfSampleResult::exercise_probabilities)
      .def_readonly("non_exercise_probability",
                    &qf::LsmOutOfSampleResult::non_exercise_probability);
  module.def("g2pp_bermudan_lsm_out_of_sample", &qf::g2pp_bermudan_lsm_out_of_sample,
             py::arg("model"), py::arg("swaption"), py::arg("config") = qf::LsmOutOfSampleConfig{});

  py::class_<qf::SwaptionQuote>(module, "SwaptionQuote")
      .def(py::init<>())
      .def_readwrite("expiry", &qf::SwaptionQuote::expiry)
      .def_readwrite("tenor", &qf::SwaptionQuote::tenor)
      .def_readwrite("strike", &qf::SwaptionQuote::strike)
      .def_readwrite("normal_volatility", &qf::SwaptionQuote::normal_volatility)
      .def_readwrite("weight", &qf::SwaptionQuote::weight);
  py::class_<qf::CalibrationDiagnostic>(module, "CalibrationDiagnostic")
      .def_readonly("expiry", &qf::CalibrationDiagnostic::expiry)
      .def_readonly("tenor", &qf::CalibrationDiagnostic::tenor)
      .def_readonly("market_volatility", &qf::CalibrationDiagnostic::market_volatility)
      .def_readonly("model_volatility", &qf::CalibrationDiagnostic::model_volatility)
      .def_readonly("error_basis_points", &qf::CalibrationDiagnostic::error_basis_points);
  py::class_<qf::G2ppCalibrationResult>(module, "G2ppCalibrationResult")
      .def_readonly("parameters", &qf::G2ppCalibrationResult::parameters)
      .def_readonly("rmse", &qf::G2ppCalibrationResult::rmse)
      .def_readonly("iterations", &qf::G2ppCalibrationResult::iterations)
      .def_readonly("converged", &qf::G2ppCalibrationResult::converged)
      .def_readonly("diagnostics", &qf::G2ppCalibrationResult::diagnostics);
  module.def(
      "calibrate_g2pp",
      [](std::shared_ptr<qf::YieldCurve> curve, const std::vector<qf::SwaptionQuote>& quotes,
         qf::G2ppParameters initial) {
        return qf::calibrate_g2pp(std::move(curve), quotes, initial);
      },
      py::arg("curve"), py::arg("quotes"), py::arg("initial") = qf::G2ppParameters{});
  py::class_<qf::G2ppMultiStartConfig>(module, "G2ppMultiStartConfig")
      .def(py::init<>())
      .def_readwrite("starts", &qf::G2ppMultiStartConfig::starts)
      .def_readwrite("seed", &qf::G2ppMultiStartConfig::seed);
  py::class_<qf::G2ppCalibrationRun>(module, "G2ppCalibrationRun")
      .def_readonly("initial", &qf::G2ppCalibrationRun::initial)
      .def_readonly("calibration", &qf::G2ppCalibrationRun::calibration);
  py::class_<qf::G2ppMultiStartResult>(module, "G2ppMultiStartResult")
      .def_readonly("best", &qf::G2ppMultiStartResult::best)
      .def_readonly("best_run", &qf::G2ppMultiStartResult::best_run)
      .def_readonly("total_iterations", &qf::G2ppMultiStartResult::total_iterations)
      .def_readonly("runs", &qf::G2ppMultiStartResult::runs);
  module.def(
      "calibrate_g2pp_multistart",
      [](std::shared_ptr<qf::YieldCurve> curve, const std::vector<qf::SwaptionQuote>& quotes,
         qf::G2ppMultiStartConfig config) {
        return qf::calibrate_g2pp_multistart(std::move(curve), quotes, config);
      },
      py::arg("curve"), py::arg("quotes"), py::arg("config") = qf::G2ppMultiStartConfig{});

  py::class_<qf::BucketedRisk>(module, "BucketedRisk")
      .def_readonly("maturity", &qf::BucketedRisk::maturity)
      .def_readonly("value", &qf::BucketedRisk::value);
  py::class_<qf::Dv01Result>(module, "Dv01Result")
      .def_readonly("base_npv", &qf::Dv01Result::base_npv)
      .def_readonly("parallel_dv01", &qf::Dv01Result::parallel_dv01)
      .def_readonly("bucketed_dv01", &qf::Dv01Result::bucketed_dv01);
  module.def("swap_dv01", &qf::swap_dv01, py::arg("swap"), py::arg("curve"),
             py::arg("bump") = 1.0e-4);

  py::class_<qf::VolatilityScenarioResult>(module, "VolatilityScenarioResult")
      .def_readonly("name", &qf::VolatilityScenarioResult::name)
      .def_readonly("volatility", &qf::VolatilityScenarioResult::volatility)
      .def_readonly("price", &qf::VolatilityScenarioResult::price)
      .def_readonly("change", &qf::VolatilityScenarioResult::change);
  module.def("run_bachelier_volatility_scenarios", &qf::run_bachelier_volatility_scenarios,
             py::arg("type"), py::arg("forward"), py::arg("strike"), py::arg("base_volatility"),
             py::arg("expiry"), py::arg("annuity"), py::arg("shock") = 0.0010,
             py::arg("notional") = 1.0);
}
