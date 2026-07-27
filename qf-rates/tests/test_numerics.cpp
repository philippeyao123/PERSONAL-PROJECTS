#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "qf/core/numerics.hpp"
#include "qf/core/random.hpp"

using Catch::Approx;

TEST_CASE("Root finding and integration converge") {
  const auto root = qf::bisection([](double x) { return x * x - 2.0; }, 0.0, 2.0);
  REQUIRE(root.converged);
  REQUIRE(root.value == Approx(std::sqrt(2.0)).margin(1.0e-9));
  REQUIRE(qf::adaptive_simpson([](double x) { return std::sin(x); }, 0.0, std::numbers::pi) ==
          Approx(2.0).margin(1.0e-10));
}

TEST_CASE("Online statistics use the unbiased sample variance") {
  qf::OnlineStatistics stats;
  for (const double value : {1.0, 2.0, 3.0, 4.0}) {
    stats.add(value);
  }
  REQUIRE(stats.mean() == Approx(2.5));
  REQUIRE(stats.variance() == Approx(5.0 / 3.0));
}

TEST_CASE("Seeded random draws are reproducible and antithetic") {
  qf::RandomEngine first(1234);
  qf::RandomEngine second(1234);
  REQUIRE(first.normal() == second.normal());
  const auto pair = first.antithetic_normal();
  REQUIRE(pair.first == Approx(-pair.second));
}

TEST_CASE("Bounded optimizer respects constraints") {
  const std::vector<double> lower{-1.0, -1.0};
  const std::vector<double> upper{1.0, 1.0};
  const auto result = qf::bounded_coordinate_search(
      [](std::span<const double> x) { return std::pow(x[0] - 0.2, 2) + std::pow(x[1] + 0.4, 2); },
      {0.9, 0.9}, lower, upper);
  REQUIRE(result.converged);
  REQUIRE(result.parameters[0] == Approx(0.2).margin(1.0e-5));
  REQUIRE(result.parameters[1] == Approx(-0.4).margin(1.0e-5));
}
