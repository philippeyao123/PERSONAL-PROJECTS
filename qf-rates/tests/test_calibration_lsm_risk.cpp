#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "qf/rates/calibration.hpp"
#include "qf/rates/lsm.hpp"
#include "qf/rates/risk.hpp"

using Catch::Approx;

TEST_CASE("Calibration diagnostics are finite and parameters stay constrained") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const std::vector<qf::SwaptionQuote> quotes{
      {1.0, 5.0, 0.026, 0.0060, 1.0},
      {2.0, 5.0, 0.026, 0.0065, 1.0},
  };
  const auto result = qf::calibrate_g2pp(curve, quotes);
  REQUIRE(std::isfinite(result.rmse));
  REQUIRE(result.parameters.a > 0.0);
  REQUIRE(result.parameters.b > 0.0);
  REQUIRE(result.parameters.rho > -1.0);
  REQUIRE(result.parameters.rho < 1.0);
  REQUIRE(result.diagnostics.size() == quotes.size());
}

TEST_CASE("One LSM exercise date reproduces the European pricer") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.02);
  const qf::G2ppModel model(curve, {});
  const qf::BermudanSwaption bermudan{{2.0}, 7.0, 0.022, 1.0, qf::OptionType::Call, 1.0};
  const auto result = qf::g2pp_bermudan_lsm(model, bermudan, {1000, 12, 42});
  const qf::EuropeanSwaption european{2.0, qf::Schedule(2.0, 7.0, 1.0), 0.022, 1.0,
                                      qf::OptionType::Call};
  REQUIRE(result.price == Approx(qf::g2pp_european_swaption(model, european)).margin(1.0e-14));
}

TEST_CASE("Bermudan value is no lower than the comparable European") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const qf::BermudanSwaption bermudan{{1.0, 2.0, 3.0}, 6.0, 0.026, 1.0, qf::OptionType::Call, 1.0};
  const auto result = qf::g2pp_bermudan_lsm(model, bermudan, {4000, 8, 123});
  const qf::EuropeanSwaption european{1.0, qf::Schedule(1.0, 6.0, 1.0), 0.026, 1.0,
                                      qf::OptionType::Call};
  REQUIRE(result.price >= qf::g2pp_european_swaption(model, european));
}

TEST_CASE("DV01 buckets reconcile with the parallel bump to first order") {
  const qf::InterpolatedYieldCurve curve({0.0, 1.0, 2.0, 5.0, 10.0},
                                         {1.0, 0.98, 0.955, 0.88, 0.75});
  const auto swap = qf::make_vanilla_swap(0.0, 10.0, 0.03, 1'000'000.0);
  const auto risk = qf::swap_dv01(swap, curve, 1.0e-5);
  double bucket_sum = 0.0;
  for (const auto& bucket : risk.bucketed_dv01) {
    bucket_sum += bucket.value;
  }
  REQUIRE(bucket_sum == Approx(risk.parallel_dv01).epsilon(0.02).margin(0.1));
}

TEST_CASE("Bucket vega aggregates to global vega") {
  const std::vector<double> vols{0.005, 0.006, 0.007};
  const std::vector<double> weights{0.2, 0.3, 0.5};
  const auto risk =
      qf::bachelier_vega_buckets(qf::OptionType::Call, 0.03, 0.03, vols, weights, 2.0, 4.5);
  double sum = 0.0;
  for (double bucket : risk.bucketed_vega) {
    sum += bucket;
  }
  REQUIRE(sum == Approx(risk.global_vega).epsilon(0.001));
}
