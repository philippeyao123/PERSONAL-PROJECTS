#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "qf/rates/g2pp.hpp"

namespace qf {

struct MonteCarloResult {
  Money price{};
  double standard_error{};
  double confidence_low{};
  double confidence_high{};
  std::size_t paths{};
};

struct MonteCarloConfig {
  std::size_t paths{50000};
  std::size_t time_steps{120};
  std::uint64_t seed{42U};
  bool antithetic{true};
};

MonteCarloResult monte_carlo_normal(const std::function<double(double)>& discounted_payoff,
                                    MonteCarloConfig config,
                                    const std::function<double(double)>& control = {},
                                    double expected_control = 0.0);

MonteCarloResult g2pp_european_swaption_mc(const G2ppModel& model, const EuropeanSwaption& swaption,
                                           MonteCarloConfig config = {});

}  // namespace qf
