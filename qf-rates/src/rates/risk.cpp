#include "qf/rates/risk.hpp"

#include <cmath>

#include "qf/core/error.hpp"

namespace qf {

Dv01Result swap_dv01(const InterestRateSwap& swap, const InterpolatedYieldCurve& curve,
                     double bump) {
  if (!(bump > 0.0)) {
    throw ValidationError("DV01 bump must be positive");
  }
  const double base = swap.npv(curve, curve);
  const auto down = curve.parallel_bumped(-bump);
  Dv01Result result{base, swap.npv(down, down) - base, {}};
  for (std::size_t bucket = 1; bucket < curve.maturities().size(); ++bucket) {
    const auto bumped = curve.bumped(-bump, bucket);
    result.bucketed_dv01.push_back({curve.maturities()[bucket], swap.npv(bumped, bumped) - base});
  }
  return result;
}

VegaResult bachelier_vega_buckets(OptionType type, Rate forward, Rate strike,
                                  std::span<const Volatility> volatility_buckets,
                                  std::span<const double> weights, Time expiry, double annuity,
                                  double bump) {
  if (volatility_buckets.empty() || volatility_buckets.size() != weights.size() || bump <= 0.0) {
    throw ValidationError("Vega buckets and weights must have equal non-zero sizes");
  }
  double aggregate_volatility = 0.0;
  double total_weight = 0.0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    if (weights[index] < 0.0 || volatility_buckets[index] < 0.0) {
      throw ValidationError("Vega weights and volatilities cannot be negative");
    }
    aggregate_volatility += weights[index] * volatility_buckets[index];
    total_weight += weights[index];
  }
  if (!(total_weight > 0.0)) {
    throw ValidationError("At least one vega weight must be positive");
  }
  aggregate_volatility /= total_weight;
  const double base =
      bachelier_swaption(type, forward, strike, aggregate_volatility, expiry, annuity);
  VegaResult result;
  result.bucketed_vega.reserve(weights.size());
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const double bumped_volatility = aggregate_volatility + bump * weights[index] / total_weight;
    result.bucketed_vega.push_back(
        bachelier_swaption(type, forward, strike, bumped_volatility, expiry, annuity) - base);
  }
  result.global_vega =
      bachelier_swaption(type, forward, strike, aggregate_volatility + bump, expiry, annuity) -
      base;
  return result;
}

std::vector<ScenarioResult> run_swap_scenarios(const InterestRateSwap& swap,
                                               const InterpolatedYieldCurve& curve, double shock) {
  if (!(shock > 0.0)) {
    throw ValidationError("Scenario shock must be positive");
  }
  const double base = swap.npv(curve, curve);
  std::vector<ScenarioResult> results;
  const auto add_parallel = [&](const std::string& name, double bump) {
    const auto scenario_curve = curve.parallel_bumped(bump);
    const double npv = swap.npv(scenario_curve, scenario_curve);
    results.push_back({name, npv, npv - base});
  };
  add_parallel("parallel_up", shock);
  add_parallel("parallel_down", -shock);
  const auto shape_scenario = [&](const std::string& name, double direction) {
    auto discounts = curve.discounts();
    const double terminal = curve.maturities().back();
    for (std::size_t index = 1; index < discounts.size(); ++index) {
      const double normalized = curve.maturities()[index] / terminal;
      const double node_shift = direction * shock * (2.0 * normalized - 1.0);
      discounts[index] *= std::exp(-node_shift * curve.maturities()[index]);
    }
    const InterpolatedYieldCurve scenario_curve(curve.maturities(), std::move(discounts),
                                                curve.interpolation());
    const double npv = swap.npv(scenario_curve, scenario_curve);
    results.push_back({name, npv, npv - base});
  };
  shape_scenario("steepener", 1.0);
  shape_scenario("flattener", -1.0);
  return results;
}

std::vector<VolatilityScenarioResult> run_bachelier_volatility_scenarios(
    OptionType type, Rate forward, Rate strike, Volatility base_volatility, Time expiry,
    double annuity, Volatility shock, Money notional) {
  if (base_volatility < 0.0 || shock <= 0.0 || expiry < 0.0 || annuity <= 0.0 || notional <= 0.0) {
    throw ValidationError("Invalid volatility scenario inputs");
  }
  const double base =
      bachelier_swaption(type, forward, strike, base_volatility, expiry, annuity, notional);
  const auto scenario = [&](const std::string& name, double volatility) {
    const double price =
        bachelier_swaption(type, forward, strike, volatility, expiry, annuity, notional);
    return VolatilityScenarioResult{name, volatility, price, price - base};
  };
  return {scenario("volatility_up", base_volatility + shock),
          scenario("volatility_down", std::max(0.0, base_volatility - shock))};
}

}  // namespace qf
