#include "qf/rates/lsm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "qf/core/error.hpp"
#include "qf/core/numerics.hpp"
#include "qf/core/random.hpp"

namespace qf {
namespace {

std::array<double, 6> basis(FactorState state) {
  return {1.0, state.x, state.y, state.x * state.x, state.x * state.y, state.y * state.y};
}

double exercise_value(const G2ppModel& model, const BermudanSwaption& swaption, Time exercise,
                      FactorState state) {
  const Schedule schedule(exercise, swaption.maturity, swaption.fixed_frequency);
  double annuity = 0.0;
  for (const auto& period : schedule.periods()) {
    annuity += period.accrual * model.discount_bond(exercise, period.payment, state);
  }
  const double payer =
      1.0 - model.discount_bond(exercise, swaption.maturity, state) - swaption.strike * annuity;
  const double signed_value = swaption.type == OptionType::Call ? payer : -payer;
  return swaption.notional * std::max(signed_value, 0.0);
}

}  // namespace

LsmResult g2pp_bermudan_lsm(const G2ppModel& model, const BermudanSwaption& swaption,
                            LsmConfig config) {
  if (swaption.exercise_times.empty() || swaption.maturity <= swaption.exercise_times.back() ||
      swaption.notional <= 0.0 || config.paths < 10U || config.steps_per_year == 0U) {
    throw ValidationError("Invalid Bermudan swaption or LSM configuration");
  }
  for (std::size_t index = 0; index < swaption.exercise_times.size(); ++index) {
    if (swaption.exercise_times[index] <= 0.0 ||
        (index > 0U && swaption.exercise_times[index] <= swaption.exercise_times[index - 1U])) {
      throw ValidationError("Exercise times must be positive and strictly increasing");
    }
  }
  if (swaption.exercise_times.size() == 1U) {
    const Time expiry = swaption.exercise_times.front();
    const EuropeanSwaption european{expiry,
                                    Schedule(expiry, swaption.maturity, swaption.fixed_frequency),
                                    swaption.strike, swaption.notional, swaption.type};
    return {g2pp_european_swaption(model, european), 0.0, {1.0}};
  }
  const std::size_t dates = swaption.exercise_times.size();
  std::vector<std::vector<FactorState>> states(dates, std::vector<FactorState>(config.paths));
  std::vector<std::vector<double>> discounts(dates, std::vector<double>(config.paths));
  RandomEngine random(config.seed);
  for (std::size_t path = 0; path < config.paths; ++path) {
    FactorState state{};
    double integral = 0.0;
    Time time = 0.0;
    for (std::size_t date = 0; date < dates; ++date) {
      const Time target = swaption.exercise_times[date];
      const std::size_t steps = std::max<std::size_t>(
          1U, static_cast<std::size_t>(std::ceil((target - time) * config.steps_per_year)));
      const double step = (target - time) / static_cast<double>(steps);
      for (std::size_t current = 0; current < steps; ++current) {
        const FactorState next = model.evolve(state, step, random.normal(), random.normal());
        integral +=
            0.5 * step * (model.short_rate(time, state) + model.short_rate(time + step, next));
        state = next;
        time += step;
      }
      states[date][path] = state;
      discounts[date][path] = std::exp(-integral);
    }
  }
  std::vector<double> values(config.paths);
  std::vector<std::size_t> exercise_index(config.paths, dates - 1U);
  for (std::size_t path = 0; path < config.paths; ++path) {
    values[path] =
        exercise_value(model, swaption, swaption.exercise_times.back(), states.back()[path]);
  }
  for (std::size_t reverse = dates - 1U; reverse-- > 0U;) {
    std::array<std::array<double, 6>, 6> normal_matrix{};
    std::array<double, 6> normal_rhs{};
    std::vector<double> immediate(config.paths);
    std::vector<double> continuation(config.paths);
    std::size_t in_the_money = 0U;
    for (std::size_t path = 0; path < config.paths; ++path) {
      immediate[path] =
          exercise_value(model, swaption, swaption.exercise_times[reverse], states[reverse][path]);
      continuation[path] = values[path] * discounts[reverse + 1U][path] / discounts[reverse][path];
      if (immediate[path] <= 0.0) {
        continue;
      }
      ++in_the_money;
      const auto regressors = basis(states[reverse][path]);
      for (std::size_t row = 0; row < regressors.size(); ++row) {
        normal_rhs[row] += regressors[row] * continuation[path];
        for (std::size_t column = 0; column < regressors.size(); ++column) {
          normal_matrix[row][column] += regressors[row] * regressors[column];
        }
      }
    }
    std::vector<double> coefficients(6U, 0.0);
    if (in_the_money >= 12U) {
      std::vector<std::vector<double>> matrix(6U, std::vector<double>(6U));
      std::vector<double> rhs(6U);
      for (std::size_t row = 0; row < 6U; ++row) {
        rhs[row] = normal_rhs[row];
        for (std::size_t column = 0; column < 6U; ++column) {
          matrix[row][column] = normal_matrix[row][column];
        }
        matrix[row][row] += 1.0e-12;
      }
      coefficients = solve_linear_system(std::move(matrix), std::move(rhs));
    }
    for (std::size_t path = 0; path < config.paths; ++path) {
      double estimated = continuation[path];
      if (immediate[path] > 0.0 && in_the_money >= 12U) {
        estimated = 0.0;
        const auto regressors = basis(states[reverse][path]);
        for (std::size_t index = 0; index < regressors.size(); ++index) {
          estimated += coefficients[index] * regressors[index];
        }
      }
      if (immediate[path] > estimated) {
        values[path] = immediate[path];
        exercise_index[path] = reverse;
      } else {
        values[path] = continuation[path];
      }
    }
  }
  OnlineStatistics statistics;
  std::vector<double> exercise_probabilities(dates, 0.0);
  for (std::size_t path = 0; path < config.paths; ++path) {
    statistics.add(values[path] * discounts.front()[path]);
    exercise_probabilities[exercise_index[path]] += 1.0 / static_cast<double>(config.paths);
  }
  const EuropeanSwaption european{
      swaption.exercise_times.front(),
      Schedule(swaption.exercise_times.front(), swaption.maturity, swaption.fixed_frequency),
      swaption.strike, swaption.notional, swaption.type};
  const double european_floor = g2pp_european_swaption(model, european);
  return {std::max(statistics.mean(), european_floor), statistics.standard_error(),
          std::move(exercise_probabilities)};
}

}  // namespace qf
