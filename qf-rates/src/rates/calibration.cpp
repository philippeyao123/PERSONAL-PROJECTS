#include "qf/rates/calibration.hpp"

#include <array>
#include <cmath>
#include <memory>

#include "qf/core/error.hpp"
#include "qf/core/numerics.hpp"
#include "qf/rates/options.hpp"
#include "qf/rates/swap.hpp"

namespace qf {
namespace {

G2ppParameters to_parameters(std::span<const double> values) {
  return {values[0], values[1], values[2], values[3], values[4]};
}

double model_normal_volatility(const G2ppModel& model, const SwaptionQuote& quote) {
  const Schedule schedule(quote.expiry, quote.expiry + quote.tenor, 1.0);
  const auto swap = make_vanilla_swap(quote.expiry, quote.expiry + quote.tenor, quote.strike, 1.0);
  const double forward = swap.par_rate(model.curve(), model.curve());
  const double annuity = swap.annuity(model.curve());
  const EuropeanSwaption swaption{quote.expiry, schedule, quote.strike, 1.0, OptionType::Call};
  const double target_price = g2pp_european_swaption(model, swaption);
  const auto objective = [&](double volatility) {
    return bachelier_swaption(OptionType::Call, forward, quote.strike, volatility, quote.expiry,
                              annuity) -
           target_price;
  };
  if (target_price <=
      bachelier_swaption(OptionType::Call, forward, quote.strike, 0.0, quote.expiry, annuity) +
          1.0e-14) {
    return 0.0;
  }
  double upper = 0.01;
  while (objective(upper) < 0.0 && upper < 1.0) {
    upper *= 2.0;
  }
  return bisection(objective, 1.0e-10, upper, 1.0e-9, 100).value;
}

}  // namespace

G2ppCalibrationResult calibrate_g2pp(YieldCurvePtr curve, std::span<const SwaptionQuote> quotes,
                                     G2ppParameters initial) {
  if (!curve || quotes.empty()) {
    throw ValidationError("Calibration requires a curve and at least one quote");
  }
  for (const auto& quote : quotes) {
    if (!(quote.expiry > 0.0 && quote.tenor > 0.0 && quote.strike >= 0.0 &&
          quote.normal_volatility >= 0.0 && quote.weight > 0.0)) {
      throw ValidationError("Calibration quote is invalid");
    }
  }
  const std::vector<double> starting{initial.a, initial.b, initial.sigma, initial.eta, initial.rho};
  const std::array<double, 5> lower{0.005, 0.01, 0.0001, 0.0001, -0.95};
  const std::array<double, 5> upper{1.00, 1.50, 0.10, 0.10, 0.95};
  const auto objective = [&](std::span<const double> values) {
    const G2ppModel model(curve, to_parameters(values));
    double squared_error = 0.0;
    double total_weight = 0.0;
    for (const auto& quote : quotes) {
      const double error = model_normal_volatility(model, quote) - quote.normal_volatility;
      squared_error += quote.weight * error * error;
      total_weight += quote.weight;
    }
    return std::sqrt(squared_error / total_weight);
  };
  const auto optimized = bounded_coordinate_search(objective, starting, lower, upper, 120, 2.0e-5);
  const auto calibrated_parameters = to_parameters(optimized.parameters);
  const G2ppModel calibrated(curve, calibrated_parameters);
  std::vector<CalibrationDiagnostic> diagnostics;
  diagnostics.reserve(quotes.size());
  for (const auto& quote : quotes) {
    const double model_volatility = model_normal_volatility(calibrated, quote);
    diagnostics.push_back({quote.expiry, quote.tenor, quote.normal_volatility, model_volatility,
                           10000.0 * (model_volatility - quote.normal_volatility)});
  }
  return {calibrated_parameters, optimized.objective, optimized.iterations, optimized.converged,
          std::move(diagnostics)};
}

}  // namespace qf
