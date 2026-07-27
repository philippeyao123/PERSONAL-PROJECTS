#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

#include "qf/rates/options.hpp"

using Catch::Approx;

TEST_CASE("Black-76 matches a known reference value") {
  const auto result = qf::black76(qf::OptionType::Call, 100.0, 100.0, 0.20, 1.0, 0.95);
  REQUIRE(result.price == Approx(7.567289).margin(1.0e-6));
  REQUIRE(result.gamma > 0.0);
  REQUIRE(result.vega > 0.0);
}

TEST_CASE("Bachelier ATM formula and Greeks are correct") {
  const auto result = qf::bachelier(qf::OptionType::Call, 0.03, 0.03, 0.01, 1.0, 0.97);
  REQUIRE(result.price == Approx(0.97 * 0.01 / std::sqrt(2.0 * std::numbers::pi)).margin(1.0e-12));
  REQUIRE(result.delta == Approx(0.485));
  REQUIRE(result.gamma > 0.0);
  REQUIRE(result.vega > 0.0);
}

TEST_CASE("Bachelier supports negative rates") {
  const auto call = qf::bachelier(qf::OptionType::Call, -0.005, -0.01, 0.006, 2.0);
  const auto put = qf::bachelier(qf::OptionType::Put, -0.005, -0.01, 0.006, 2.0);
  REQUIRE(call.price > 0.0);
  REQUIRE(put.price > 0.0);
  REQUIRE(call.price - put.price == Approx(0.005).margin(1.0e-12));
}

TEST_CASE("Put-call parity and monotonicity hold") {
  const auto call = qf::black76(qf::OptionType::Call, 0.035, 0.03, 0.20, 2.0, 0.95);
  const auto put = qf::black76(qf::OptionType::Put, 0.035, 0.03, 0.20, 2.0, 0.95);
  REQUIRE(call.price - put.price == Approx(0.95 * (0.035 - 0.03)).margin(1.0e-12));
  REQUIRE(qf::black76(qf::OptionType::Call, 0.04, 0.03, 0.20, 2.0).price >
          qf::black76(qf::OptionType::Call, 0.035, 0.03, 0.20, 2.0).price);
  REQUIRE(call.price >= 0.0);
  REQUIRE(put.price >= 0.0);
}
