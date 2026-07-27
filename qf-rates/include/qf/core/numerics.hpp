#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace qf {

struct SolverResult {
  double value{};
  std::size_t iterations{};
  bool converged{};
};

SolverResult bisection(const std::function<double(double)>& function, double lower, double upper,
                       double tolerance = 1.0e-10, std::size_t max_iterations = 200);

double adaptive_simpson(const std::function<double(double)>& function, double lower, double upper,
                        double tolerance = 1.0e-9, std::size_t max_depth = 20);

std::vector<double> solve_linear_system(std::vector<std::vector<double>> matrix,
                                        std::vector<double> rhs);

class OnlineStatistics {
 public:
  void add(double value) noexcept;
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] double mean() const noexcept { return mean_; }
  [[nodiscard]] double variance() const noexcept;
  [[nodiscard]] double standard_deviation() const noexcept;
  [[nodiscard]] double standard_error() const noexcept;

 private:
  std::size_t count_{};
  double mean_{};
  double m2_{};
};

struct OptimizerResult {
  std::vector<double> parameters;
  double objective{};
  std::size_t iterations{};
  bool converged{};
};

OptimizerResult bounded_coordinate_search(
    const std::function<double(std::span<const double>)>& objective, std::vector<double> initial,
    std::span<const double> lower, std::span<const double> upper, std::size_t max_iterations = 250,
    double tolerance = 1.0e-7);

double normal_pdf(double x) noexcept;
double normal_cdf(double x) noexcept;

}  // namespace qf
