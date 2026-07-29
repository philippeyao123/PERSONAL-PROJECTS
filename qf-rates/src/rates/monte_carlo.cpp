#include "qf/rates/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "qf/core/error.hpp"
#include "qf/core/numerics.hpp"
#include "qf/core/random.hpp"

namespace qf {
namespace {

double swaption_payoff(const G2ppModel& model, const EuropeanSwaption& swaption,
                       FactorState state) {
  const auto& periods = swaption.underlying_schedule.periods();
  double fixed_annuity = 0.0;
  for (const auto& period : periods) {
    fixed_annuity += period.accrual * model.discount_bond(swaption.expiry, period.payment, state);
  }
  const double payer = 1.0 - model.discount_bond(swaption.expiry, periods.back().end, state) -
                       swaption.strike * fixed_annuity;
  const double signed_value = swaption.type == OptionType::Call ? payer : -payer;
  return swaption.notional * std::max(signed_value, 0.0);
}

MonteCarloResult result_from_statistics(const OnlineStatistics& statistics,
                                        std::size_t reported_paths) {
  constexpr double critical_value = 1.959963984540054;
  const double error = statistics.standard_error();
  return {statistics.mean(), error, statistics.mean() - critical_value * error,
          statistics.mean() + critical_value * error, reported_paths};
}

}  // namespace

MonteCarloResult monte_carlo_normal(const std::function<double(double)>& discounted_payoff,
                                    MonteCarloConfig config,
                                    const std::function<double(double)>& control,
                                    double expected_control) {
  if (config.paths < 2U) {
    throw ValidationError("Monte-Carlo requires at least two paths");
  }
  RandomEngine random(config.seed);
  std::vector<double> payoffs;
  std::vector<double> controls;
  const std::size_t observations = config.antithetic ? (config.paths + 1U) / 2U : config.paths;
  payoffs.reserve(observations);
  controls.reserve(observations);
  std::size_t simulated_paths = 0U;
  while (payoffs.size() < observations) {
    const double normal = random.normal();
    if (config.antithetic && simulated_paths + 1U < config.paths) {
      payoffs.push_back(0.5 * (discounted_payoff(normal) + discounted_payoff(-normal)));
      if (control) {
        controls.push_back(0.5 * (control(normal) + control(-normal)));
      }
      simulated_paths += 2U;
    } else {
      payoffs.push_back(discounted_payoff(normal));
      if (control) {
        controls.push_back(control(normal));
      }
      ++simulated_paths;
    }
  }
  double beta = 0.0;
  if (control) {
    double mean_payoff = 0.0;
    double mean_control = 0.0;
    for (std::size_t index = 0; index < payoffs.size(); ++index) {
      mean_payoff += payoffs[index];
      mean_control += controls[index];
    }
    mean_payoff /= static_cast<double>(payoffs.size());
    mean_control /= static_cast<double>(controls.size());
    double covariance = 0.0;
    double variance = 0.0;
    for (std::size_t index = 0; index < payoffs.size(); ++index) {
      covariance += (payoffs[index] - mean_payoff) * (controls[index] - mean_control);
      variance += (controls[index] - mean_control) * (controls[index] - mean_control);
    }
    beta = variance > 0.0 ? covariance / variance : 0.0;
  }
  OnlineStatistics statistics;
  for (std::size_t index = 0; index < payoffs.size(); ++index) {
    const double adjusted =
        control ? payoffs[index] - beta * (controls[index] - expected_control) : payoffs[index];
    statistics.add(adjusted);
  }
  return result_from_statistics(statistics, config.paths);
}

MonteCarloResult g2pp_european_swaption_mc(const G2ppModel& model, const EuropeanSwaption& swaption,
                                           MonteCarloConfig config) {
  if (config.paths < 2U || config.time_steps == 0U || swaption.expiry <= 0.0) {
    throw ValidationError("Invalid G2++ Monte-Carlo configuration");
  }
  RandomEngine random(config.seed);
  OnlineStatistics statistics;
  const double step = swaption.expiry / static_cast<double>(config.time_steps);
  const std::size_t pairs = config.antithetic ? (config.paths + 1U) / 2U : config.paths;
  std::size_t consumed = 0U;
  for (std::size_t path = 0; path < pairs; ++path) {
    FactorState state{};
    FactorState opposite{};
    double integral = 0.0;
    double opposite_integral = 0.0;
    for (std::size_t time_index = 0; time_index < config.time_steps; ++time_index) {
      const Time time = static_cast<double>(time_index) * step;
      const double first = random.normal();
      const double second = random.normal();
      const FactorState next = model.evolve(state, step, first, second);
      const FactorState opposite_next = model.evolve(opposite, step, -first, -second);
      integral +=
          0.5 * step * (model.short_rate(time, state) + model.short_rate(time + step, next));
      opposite_integral +=
          0.5 * step *
          (model.short_rate(time, opposite) + model.short_rate(time + step, opposite_next));
      state = next;
      opposite = opposite_next;
    }
    const double first_value = std::exp(-integral) * swaption_payoff(model, swaption, state);
    if (config.antithetic) {
      const double second_value =
          std::exp(-opposite_integral) * swaption_payoff(model, swaption, opposite);
      statistics.add(0.5 * (first_value + second_value));
      consumed += std::min<std::size_t>(2U, config.paths - consumed);
    } else {
      statistics.add(first_value);
      ++consumed;
    }
  }
  auto result = result_from_statistics(statistics, config.paths);
  if (config.antithetic) {
    result.standard_error = statistics.standard_error();
    constexpr double critical_value = 1.959963984540054;
    result.confidence_low = result.price - critical_value * result.standard_error;
    result.confidence_high = result.price + critical_value * result.standard_error;
  }
  return result;
}

std::vector<MonteCarloTimeConvergenceResult> g2pp_european_swaption_mc_time_convergence(
    const G2ppModel& model, const EuropeanSwaption& swaption,
    MonteCarloTimeConvergenceConfig config) {
  if (config.paths < 2U || config.finest_time_steps == 0U || swaption.expiry <= 0.0 ||
      config.time_steps.empty()) {
    throw ValidationError("Invalid G2++ time-convergence configuration");
  }
  if (config.antithetic && config.paths % 2U != 0U) {
    throw ValidationError("Antithetic time convergence requires an even path count");
  }
  for (std::size_t index = 0; index < config.time_steps.size(); ++index) {
    const std::size_t current = config.time_steps[index];
    if (current == 0U || config.finest_time_steps % current != 0U ||
        (index > 0U && current <= config.time_steps[index - 1U])) {
      throw ValidationError(
          "Time-convergence levels must be increasing divisors of the finest grid");
    }
  }
  if (config.time_steps.back() != config.finest_time_steps) {
    throw ValidationError("Time-convergence levels must include the finest grid last");
  }

  const std::size_t levels = config.time_steps.size();
  std::vector<std::size_t> strides(levels);
  for (std::size_t level = 0; level < levels; ++level) {
    strides[level] = config.finest_time_steps / config.time_steps[level];
  }
  std::vector<OnlineStatistics> price_statistics(levels);
  std::vector<OnlineStatistics> difference_statistics(levels);
  RandomEngine random(config.seed);
  const double fine_step = swaption.expiry / static_cast<double>(config.finest_time_steps);
  const std::size_t observations = config.antithetic ? config.paths / 2U : config.paths;
  for (std::size_t path = 0; path < observations; ++path) {
    FactorState state{};
    FactorState opposite{};
    const double initial_rate = model.short_rate(0.0, {});
    std::vector<double> integrals(levels, 0.0);
    std::vector<double> opposite_integrals(levels, 0.0);
    std::vector<double> previous_rates(levels, initial_rate);
    std::vector<double> opposite_previous_rates(levels, initial_rate);
    for (std::size_t fine_index = 0; fine_index < config.finest_time_steps; ++fine_index) {
      const double first = random.normal();
      const double second = random.normal();
      const FactorState next = model.evolve(state, fine_step, first, second);
      const FactorState opposite_next = model.evolve(opposite, fine_step, -first, -second);
      const Time time = static_cast<double>(fine_index + 1U) * fine_step;
      const double current_rate = model.short_rate(time, next);
      const double opposite_current_rate = model.short_rate(time, opposite_next);
      for (std::size_t level = 0; level < levels; ++level) {
        if ((fine_index + 1U) % strides[level] != 0U) {
          continue;
        }
        const double coarse_step = fine_step * static_cast<double>(strides[level]);
        integrals[level] += 0.5 * coarse_step * (previous_rates[level] + current_rate);
        opposite_integrals[level] +=
            0.5 * coarse_step * (opposite_previous_rates[level] + opposite_current_rate);
        previous_rates[level] = current_rate;
        opposite_previous_rates[level] = opposite_current_rate;
      }
      state = next;
      opposite = opposite_next;
    }
    const double payoff = swaption_payoff(model, swaption, state);
    const double opposite_payoff = swaption_payoff(model, swaption, opposite);
    std::vector<double> values(levels);
    for (std::size_t level = 0; level < levels; ++level) {
      values[level] = std::exp(-integrals[level]) * payoff;
      if (config.antithetic) {
        values[level] =
            0.5 * (values[level] + std::exp(-opposite_integrals[level]) * opposite_payoff);
      }
    }
    const double finest_value = values.back();
    for (std::size_t level = 0; level < levels; ++level) {
      price_statistics[level].add(values[level]);
      difference_statistics[level].add(values[level] - finest_value);
    }
  }

  std::vector<MonteCarloTimeConvergenceResult> results;
  results.reserve(levels);
  for (std::size_t level = 0; level < levels; ++level) {
    results.push_back({config.time_steps[level], price_statistics[level].mean(),
                       price_statistics[level].standard_error(),
                       difference_statistics[level].mean(),
                       difference_statistics[level].standard_error()});
  }
  return results;
}

}  // namespace qf
