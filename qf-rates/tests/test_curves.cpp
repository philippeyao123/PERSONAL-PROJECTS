#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "qf/core/error.hpp"
#include "qf/rates/yield_curve.hpp"

using Catch::Approx;

TEST_CASE("Flat curve reproduces continuous compounding") {
  const qf::FlatYieldCurve curve(0.03);
  REQUIRE(curve.discount(0.0) == Approx(1.0));
  REQUIRE(curve.discount(5.0) == Approx(std::exp(-0.15)));
  REQUIRE(curve.zero_rate(7.0) == Approx(0.03));
  REQUIRE(curve.forward_rate(2.0, 8.0) == Approx(0.03));
}

TEST_CASE("Interpolated curve reproduces nodes and stays positive") {
  const std::vector<double> times{0.0, 1.0, 2.0, 5.0};
  const std::vector<double> discounts{1.0, 0.98, 0.95, 0.86};
  for (const auto method :
       {qf::Interpolation::LinearDiscount, qf::Interpolation::LogLinearDiscount}) {
    const qf::InterpolatedYieldCurve curve(times, discounts, method);
    for (std::size_t index = 0; index < times.size(); ++index) {
      REQUIRE(curve.discount(times[index]) == Approx(discounts[index]).margin(1.0e-14));
    }
    REQUIRE(curve.discount(1.5) > 0.0);
    REQUIRE(curve.discount(8.0) > 0.0);
  }
}

TEST_CASE("Invalid curve inputs fail with actionable exceptions") {
  REQUIRE_THROWS_AS(qf::InterpolatedYieldCurve({0.0, 2.0, 1.0}, {1.0, 0.95, 0.97}),
                    qf::ValidationError);
  REQUIRE_THROWS_AS(qf::InterpolatedYieldCurve({0.0, 1.0}, {1.0, -0.5}), qf::ValidationError);
}
