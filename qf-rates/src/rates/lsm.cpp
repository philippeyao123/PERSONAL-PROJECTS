#include "qf/rates/lsm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "qf/core/error.hpp"
#include "qf/core/numerics.hpp"
#include "qf/core/random.hpp"

namespace qf {
namespace {

std::vector<double> basis(FactorState state, LsmBasis choice) {
  if (choice == LsmBasis::Linear) {
    return {1.0, state.x, state.y};
  }
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

void validate_lsm_inputs(const BermudanSwaption& swaption, std::size_t paths,
                         std::size_t steps_per_year) {
  if (swaption.exercise_times.empty() || swaption.maturity <= swaption.exercise_times.back() ||
      swaption.notional <= 0.0 || paths < 10U || steps_per_year == 0U) {
    throw ValidationError("Invalid Bermudan swaption or LSM configuration");
  }
  for (std::size_t index = 0; index < swaption.exercise_times.size(); ++index) {
    if (swaption.exercise_times[index] <= 0.0 ||
        (index > 0U && swaption.exercise_times[index] <= swaption.exercise_times[index - 1U])) {
      throw ValidationError("Exercise times must be positive and strictly increasing");
    }
  }
}

struct LsmPaths {
  std::vector<std::vector<FactorState>> states;
  std::vector<std::vector<double>> discounts;
};

LsmPaths simulate_lsm_paths(const G2ppModel& model, const BermudanSwaption& swaption,
                            std::size_t paths, std::size_t steps_per_year, std::uint64_t seed) {
  const std::size_t dates = swaption.exercise_times.size();
  LsmPaths simulation{std::vector<std::vector<FactorState>>(dates, std::vector<FactorState>(paths)),
                      std::vector<std::vector<double>>(dates, std::vector<double>(paths))};
  RandomEngine random(seed);
  for (std::size_t path = 0; path < paths; ++path) {
    FactorState state{};
    double integral = 0.0;
    Time time = 0.0;
    for (std::size_t date = 0; date < dates; ++date) {
      const Time target = swaption.exercise_times[date];
      const std::size_t steps = std::max<std::size_t>(
          1U, static_cast<std::size_t>(std::ceil((target - time) * steps_per_year)));
      const double step = (target - time) / static_cast<double>(steps);
      for (std::size_t current = 0; current < steps; ++current) {
        const FactorState next = model.evolve(state, step, random.normal(), random.normal());
        integral +=
            0.5 * step * (model.short_rate(time, state) + model.short_rate(time + step, next));
        state = next;
        time += step;
      }
      simulation.states[date][path] = state;
      simulation.discounts[date][path] = std::exp(-integral);
    }
  }
  return simulation;
}

struct FittedLsmPolicy {
  std::vector<std::vector<double>> coefficients;
  std::vector<bool> fitted;
  double training_price{};
};

FittedLsmPolicy fit_lsm_policy(const G2ppModel& model, const BermudanSwaption& swaption,
                               const LsmPaths& paths, LsmBasis basis_choice) {
  const std::size_t dates = swaption.exercise_times.size();
  const std::size_t path_count = paths.states.front().size();
  const std::size_t basis_size = basis_choice == LsmBasis::Linear ? 3U : 6U;
  FittedLsmPolicy policy{
      std::vector<std::vector<double>>(dates, std::vector<double>(basis_size, 0.0)),
      std::vector<bool>(dates, false), 0.0};
  std::vector<double> values(path_count);
  for (std::size_t path = 0; path < path_count; ++path) {
    values[path] =
        exercise_value(model, swaption, swaption.exercise_times.back(), paths.states.back()[path]);
  }
  for (std::size_t reverse = dates - 1U; reverse-- > 0U;) {
    std::vector<std::vector<double>> matrix(basis_size, std::vector<double>(basis_size));
    std::vector<double> rhs(basis_size);
    std::vector<double> immediate(path_count);
    std::vector<double> continuation(path_count);
    std::size_t in_the_money = 0U;
    for (std::size_t path = 0; path < path_count; ++path) {
      immediate[path] = exercise_value(model, swaption, swaption.exercise_times[reverse],
                                       paths.states[reverse][path]);
      continuation[path] =
          values[path] * paths.discounts[reverse + 1U][path] / paths.discounts[reverse][path];
      if (immediate[path] <= 0.0) {
        continue;
      }
      ++in_the_money;
      const auto regressors = basis(paths.states[reverse][path], basis_choice);
      for (std::size_t row = 0; row < basis_size; ++row) {
        rhs[row] += regressors[row] * continuation[path];
        for (std::size_t column = 0; column < basis_size; ++column) {
          matrix[row][column] += regressors[row] * regressors[column];
        }
      }
    }
    const std::size_t minimum_regression_paths = 2U * basis_size;
    if (in_the_money >= minimum_regression_paths) {
      for (std::size_t row = 0; row < basis_size; ++row) {
        matrix[row][row] += 1.0e-12;
      }
      policy.coefficients[reverse] = solve_linear_system(std::move(matrix), std::move(rhs));
      policy.fitted[reverse] = true;
    }
    for (std::size_t path = 0; path < path_count; ++path) {
      double estimated = std::numeric_limits<double>::infinity();
      if (immediate[path] > 0.0 && policy.fitted[reverse]) {
        estimated = 0.0;
        const auto regressors = basis(paths.states[reverse][path], basis_choice);
        for (std::size_t index = 0; index < basis_size; ++index) {
          estimated += policy.coefficients[reverse][index] * regressors[index];
        }
      }
      if (immediate[path] > estimated) {
        values[path] = immediate[path];
      } else {
        values[path] = continuation[path];
      }
    }
  }
  OnlineStatistics statistics;
  for (std::size_t path = 0; path < path_count; ++path) {
    statistics.add(values[path] * paths.discounts.front()[path]);
  }
  policy.training_price = statistics.mean();
  return policy;
}

LsmOutOfSampleResult evaluate_lsm_policy(const G2ppModel& model, const BermudanSwaption& swaption,
                                         const LsmPaths& paths, LsmBasis basis_choice,
                                         const FittedLsmPolicy& policy) {
  const std::size_t dates = swaption.exercise_times.size();
  const std::size_t path_count = paths.states.front().size();
  OnlineStatistics statistics;
  std::vector<double> probabilities(dates, 0.0);
  std::size_t non_exercise = 0U;
  for (std::size_t path = 0; path < path_count; ++path) {
    double present_value = 0.0;
    bool exercised = false;
    for (std::size_t date = 0; date < dates; ++date) {
      const double immediate =
          exercise_value(model, swaption, swaption.exercise_times[date], paths.states[date][path]);
      bool exercise = date + 1U == dates && immediate > 0.0;
      if (date + 1U < dates && immediate > 0.0 && policy.fitted[date]) {
        double continuation = 0.0;
        const auto regressors = basis(paths.states[date][path], basis_choice);
        for (std::size_t index = 0; index < regressors.size(); ++index) {
          continuation += policy.coefficients[date][index] * regressors[index];
        }
        exercise = immediate > continuation;
      }
      if (exercise) {
        present_value = immediate * paths.discounts[date][path];
        probabilities[date] += 1.0 / static_cast<double>(path_count);
        exercised = true;
        break;
      }
    }
    if (!exercised) {
      ++non_exercise;
    }
    statistics.add(present_value);
  }
  return {policy.training_price, statistics.mean(), statistics.standard_error(),
          std::move(probabilities),
          static_cast<double>(non_exercise) / static_cast<double>(path_count)};
}

}  // namespace

LsmResult g2pp_bermudan_lsm(const G2ppModel& model, const BermudanSwaption& swaption,
                            LsmConfig config) {
  validate_lsm_inputs(swaption, config.paths, config.steps_per_year);
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
    const std::size_t basis_size =
        config.basis == LsmBasis::Linear ? std::size_t{3U} : std::size_t{6U};
    std::vector<std::vector<double>> normal_matrix(basis_size, std::vector<double>(basis_size));
    std::vector<double> normal_rhs(basis_size);
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
      const auto regressors = basis(states[reverse][path], config.basis);
      for (std::size_t row = 0; row < regressors.size(); ++row) {
        normal_rhs[row] += regressors[row] * continuation[path];
        for (std::size_t column = 0; column < regressors.size(); ++column) {
          normal_matrix[row][column] += regressors[row] * regressors[column];
        }
      }
    }
    std::vector<double> coefficients(basis_size, 0.0);
    const std::size_t minimum_regression_paths = 2U * basis_size;
    if (in_the_money >= minimum_regression_paths) {
      std::vector<std::vector<double>> matrix(basis_size, std::vector<double>(basis_size));
      std::vector<double> rhs(basis_size);
      for (std::size_t row = 0; row < basis_size; ++row) {
        rhs[row] = normal_rhs[row];
        for (std::size_t column = 0; column < basis_size; ++column) {
          matrix[row][column] = normal_matrix[row][column];
        }
        matrix[row][row] += 1.0e-12;
      }
      coefficients = solve_linear_system(std::move(matrix), std::move(rhs));
    }
    for (std::size_t path = 0; path < config.paths; ++path) {
      double estimated = continuation[path];
      if (immediate[path] > 0.0 && in_the_money >= minimum_regression_paths) {
        estimated = 0.0;
        const auto regressors = basis(states[reverse][path], config.basis);
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

LsmOutOfSampleResult g2pp_bermudan_lsm_out_of_sample(const G2ppModel& model,
                                                     const BermudanSwaption& swaption,
                                                     LsmOutOfSampleConfig config) {
  validate_lsm_inputs(swaption, config.training_paths, config.steps_per_year);
  validate_lsm_inputs(swaption, config.valuation_paths, config.steps_per_year);
  if (config.training_seed == config.valuation_seed) {
    throw ValidationError("Out-of-sample LSM requires distinct training and valuation seeds");
  }
  if (swaption.exercise_times.size() == 1U) {
    const Time expiry = swaption.exercise_times.front();
    const EuropeanSwaption european{expiry,
                                    Schedule(expiry, swaption.maturity, swaption.fixed_frequency),
                                    swaption.strike, swaption.notional, swaption.type};
    const double price = g2pp_european_swaption(model, european);
    return {price, price, 0.0, {1.0}, 0.0};
  }
  const auto training = simulate_lsm_paths(model, swaption, config.training_paths,
                                           config.steps_per_year, config.training_seed);
  const auto policy = fit_lsm_policy(model, swaption, training, config.basis);
  const auto valuation = simulate_lsm_paths(model, swaption, config.valuation_paths,
                                            config.steps_per_year, config.valuation_seed);
  return evaluate_lsm_policy(model, swaption, valuation, config.basis, policy);
}

}  // namespace qf
