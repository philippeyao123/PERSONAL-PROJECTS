#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "qf/rates/g2pp.hpp"
#include "qf/rates/options.hpp"
#include "qf/rates/swap.hpp"

using Catch::Approx;

TEST_CASE("G2++ exactly fits the initial discount curve") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  for (const double maturity : {0.25, 1.0, 5.0, 20.0}) {
    REQUIRE(model.discount_bond(0.0, maturity) ==
            Approx(curve->discount(maturity)).margin(1.0e-14));
  }
}

TEST_CASE("G2++ covariance is positive semidefinite") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.02);
  const qf::G2ppModel model(curve, {0.10, 0.30, 0.01, 0.015, -0.70});
  const auto covariance = model.factor_covariance(5.0);
  REQUIRE(covariance[0][0] >= 0.0);
  REQUIRE(covariance[1][1] >= 0.0);
  REQUIRE(covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0] >= -1.0e-15);
  REQUIRE(model.integrated_variance(5.0) >= 0.0);
}

TEST_CASE("European G2++ swaption is positive and bounded by notional") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.03);
  const qf::G2ppModel model(curve, {});
  const auto swap = qf::make_vanilla_swap(2.0, 7.0, 0.03);
  const qf::EuropeanSwaption option{2.0, qf::Schedule(2.0, 7.0, 1.0), swap.par_rate(*curve, *curve),
                                    1.0, qf::OptionType::Call};
  const double price = qf::g2pp_european_swaption(model, option);
  REQUIRE(price > 0.0);
  REQUIRE(price < 1.0);
}
