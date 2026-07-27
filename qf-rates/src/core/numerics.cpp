#include "qf/core/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef QF_HAS_EIGEN
#include <Eigen/Dense>
#endif

#include "qf/core/error.hpp"

namespace qf {
namespace {

double simpson(const std::function<double(double)>& function, double lower, double upper) {
  const double middle = 0.5 * (lower + upper);
  return (upper - lower) * (function(lower) + 4.0 * function(middle) + function(upper)) / 6.0;
}

double adaptive_simpson_impl(const std::function<double(double)>& function, double lower,
                             double upper, double whole, double tolerance, std::size_t depth) {
  const double middle = 0.5 * (lower + upper);
  const double left = simpson(function, lower, middle);
  const double right = simpson(function, middle, upper);
  const double delta = left + right - whole;
  if (depth == 0U || std::abs(delta) <= 15.0 * tolerance) {
    return left + right + delta / 15.0;
  }
  return adaptive_simpson_impl(function, lower, middle, left, tolerance * 0.5, depth - 1U) +
         adaptive_simpson_impl(function, middle, upper, right, tolerance * 0.5, depth - 1U);
}

}  // namespace

SolverResult bisection(const std::function<double(double)>& function, double lower, double upper,
                       double tolerance, std::size_t max_iterations) {
  if (!(lower < upper && tolerance > 0.0 && max_iterations > 0U)) {
    throw ValidationError("Invalid bisection configuration");
  }
  double f_lower = function(lower);
  const double f_upper = function(upper);
  if (!std::isfinite(f_lower) || !std::isfinite(f_upper) || f_lower * f_upper > 0.0) {
    throw NumericalError("Bisection interval does not bracket a finite root");
  }
  for (std::size_t iteration = 1; iteration <= max_iterations; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    const double f_middle = function(middle);
    if (!std::isfinite(f_middle)) {
      throw NumericalError("Bisection function returned a non-finite value");
    }
    if (std::abs(f_middle) <= tolerance || 0.5 * (upper - lower) <= tolerance) {
      return {middle, iteration, true};
    }
    if (f_lower * f_middle <= 0.0) {
      upper = middle;
    } else {
      lower = middle;
      f_lower = f_middle;
    }
  }
  return {0.5 * (lower + upper), max_iterations, false};
}

double adaptive_simpson(const std::function<double(double)>& function, double lower, double upper,
                        double tolerance, std::size_t max_depth) {
  if (!(lower < upper && tolerance > 0.0 && max_depth > 0U)) {
    throw ValidationError("Invalid adaptive Simpson configuration");
  }
  return adaptive_simpson_impl(function, lower, upper, simpson(function, lower, upper), tolerance,
                               max_depth);
}

std::vector<double> solve_linear_system(std::vector<std::vector<double>> matrix,
                                        std::vector<double> rhs) {
  const std::size_t size = matrix.size();
  if (size == 0U || rhs.size() != size) {
    throw ValidationError("Linear system has incompatible dimensions");
  }
  for (const auto& row : matrix) {
    if (row.size() != size) {
      throw ValidationError("Linear system matrix must be square");
    }
  }
#ifdef QF_HAS_EIGEN
  Eigen::MatrixXd eigen_matrix(static_cast<Eigen::Index>(size), static_cast<Eigen::Index>(size));
  Eigen::VectorXd eigen_rhs(static_cast<Eigen::Index>(size));
  for (std::size_t row = 0; row < size; ++row) {
    eigen_rhs[static_cast<Eigen::Index>(row)] = rhs[row];
    for (std::size_t column = 0; column < size; ++column) {
      eigen_matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
          matrix[row][column];
    }
  }
  const Eigen::FullPivLU<Eigen::MatrixXd> decomposition(eigen_matrix);
  if (!decomposition.isInvertible()) {
    throw NumericalError("Singular linear system");
  }
  const Eigen::VectorXd eigen_solution = decomposition.solve(eigen_rhs);
  std::vector<double> solution(size);
  for (std::size_t index = 0; index < size; ++index) {
    solution[index] = eigen_solution[static_cast<Eigen::Index>(index)];
  }
  return solution;
#else
  for (std::size_t column = 0; column < size; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1U; row < size; ++row) {
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    if (std::abs(matrix[pivot][column]) < 1.0e-14) {
      throw NumericalError("Singular linear system");
    }
    std::swap(matrix[column], matrix[pivot]);
    std::swap(rhs[column], rhs[pivot]);
    const double diagonal = matrix[column][column];
    for (std::size_t entry = column; entry < size; ++entry) {
      matrix[column][entry] /= diagonal;
    }
    rhs[column] /= diagonal;
    for (std::size_t row = 0; row < size; ++row) {
      if (row == column) {
        continue;
      }
      const double factor = matrix[row][column];
      for (std::size_t entry = column; entry < size; ++entry) {
        matrix[row][entry] -= factor * matrix[column][entry];
      }
      rhs[row] -= factor * rhs[column];
    }
  }
  return rhs;
#endif
}

void OnlineStatistics::add(double value) noexcept {
  ++count_;
  const double delta = value - mean_;
  mean_ += delta / static_cast<double>(count_);
  m2_ += delta * (value - mean_);
}

double OnlineStatistics::variance() const noexcept {
  return count_ > 1U ? m2_ / static_cast<double>(count_ - 1U) : 0.0;
}

double OnlineStatistics::standard_deviation() const noexcept { return std::sqrt(variance()); }

double OnlineStatistics::standard_error() const noexcept {
  return count_ > 0U ? standard_deviation() / std::sqrt(static_cast<double>(count_)) : 0.0;
}

OptimizerResult bounded_coordinate_search(
    const std::function<double(std::span<const double>)>& objective, std::vector<double> initial,
    std::span<const double> lower, std::span<const double> upper, std::size_t max_iterations,
    double tolerance) {
  if (initial.empty() || initial.size() != lower.size() || initial.size() != upper.size()) {
    throw ValidationError("Optimizer dimensions are inconsistent");
  }
  std::vector<double> step(initial.size());
  for (std::size_t index = 0; index < initial.size(); ++index) {
    if (!(lower[index] < upper[index])) {
      throw ValidationError("Optimizer bounds are invalid");
    }
    initial[index] = std::clamp(initial[index], lower[index], upper[index]);
    step[index] = 0.2 * (upper[index] - lower[index]);
  }
  double best = objective(initial);
  for (std::size_t iteration = 1; iteration <= max_iterations; ++iteration) {
    bool improved = false;
    for (std::size_t index = 0; index < initial.size(); ++index) {
      for (const double direction : {-1.0, 1.0}) {
        auto candidate = initial;
        candidate[index] =
            std::clamp(candidate[index] + direction * step[index], lower[index], upper[index]);
        const double value = objective(candidate);
        if (value < best) {
          initial = std::move(candidate);
          best = value;
          improved = true;
        }
      }
    }
    if (!improved) {
      for (double& current_step : step) {
        current_step *= 0.5;
      }
    }
    if (*std::max_element(step.begin(), step.end()) < tolerance) {
      return {initial, best, iteration, true};
    }
  }
  return {initial, best, max_iterations, false};
}

double normal_pdf(double x) noexcept {
  constexpr double inverse_sqrt_two_pi = 0.39894228040143267794;
  return inverse_sqrt_two_pi * std::exp(-0.5 * x * x);
}

double normal_cdf(double x) noexcept { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

}  // namespace qf
