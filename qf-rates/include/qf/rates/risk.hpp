#pragma once

#include <string>
#include <vector>

#include "qf/rates/options.hpp"
#include "qf/rates/swap.hpp"

namespace qf {

struct BucketedRisk {
  Time maturity{};
  Money value{};
};

struct Dv01Result {
  Money base_npv{};
  Money parallel_dv01{};
  std::vector<BucketedRisk> bucketed_dv01;
};

Dv01Result swap_dv01(const InterestRateSwap& swap, const InterpolatedYieldCurve& curve,
                     double bump = 1.0e-4);

struct VegaResult {
  double global_vega{};
  std::vector<double> bucketed_vega;
};

VegaResult bachelier_vega_buckets(OptionType type, Rate forward, Rate strike,
                                  std::span<const Volatility> volatility_buckets,
                                  std::span<const double> weights, Time expiry, double annuity,
                                  double bump = 1.0e-4);

enum class CurveScenario { ParallelUp, ParallelDown, Steepener, Flattener };

struct ScenarioResult {
  std::string name;
  Money npv{};
  Money change{};
};

std::vector<ScenarioResult> run_swap_scenarios(const InterestRateSwap& swap,
                                               const InterpolatedYieldCurve& curve,
                                               double shock = 0.0010);

}  // namespace qf
