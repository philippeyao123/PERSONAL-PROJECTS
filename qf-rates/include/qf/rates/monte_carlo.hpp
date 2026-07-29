#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

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

struct MonteCarloTimeConvergenceConfig {
  std::size_t paths{20000};
  std::size_t finest_time_steps{384};
  std::vector<std::size_t> time_steps{12, 24, 48, 96, 192, 384};
  std::uint64_t seed{42U};
  bool antithetic{true};
};

struct MonteCarloTimeConvergenceResult {
  std::size_t time_steps{};
  Money price{};
  double standard_error{};
  double paired_bias_vs_finest{};
  double paired_bias_standard_error{};
};

MonteCarloResult monte_carlo_normal(const std::function<double(double)>& discounted_payoff,
                                    MonteCarloConfig config,
                                    const std::function<double(double)>& control = {},
                                    double expected_control = 0.0);

MonteCarloResult g2pp_european_swaption_mc(const G2ppModel& model, const EuropeanSwaption& swaption,
                                           MonteCarloConfig config = {});

std::vector<MonteCarloTimeConvergenceResult> g2pp_european_swaption_mc_time_convergence(
    const G2ppModel& model, const EuropeanSwaption& swaption,
    MonteCarloTimeConvergenceConfig config = {});

}  // namespace qf
