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

TEST_CASE("LSM convergence grid is reproducible across paths seeds bases and dates") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const std::vector<qf::BermudanSwaption> date_grids{
      {{1.0, 2.0}, 6.0, 0.026, 1.0, qf::OptionType::Call, 1.0},
      {{1.0, 2.0, 3.0}, 6.0, 0.026, 1.0, qf::OptionType::Call, 1.0},
  };
  for (const auto& swaption : date_grids) {
    for (const auto basis : {qf::LsmBasis::Linear, qf::LsmBasis::Quadratic}) {
      for (const auto seed : {7U, 42U}) {
        const qf::LsmConfig config{2000, 8, seed, basis};
        const auto first = qf::g2pp_bermudan_lsm(model, swaption, config);
        const auto repeated = qf::g2pp_bermudan_lsm(model, swaption, config);
        REQUIRE(first.price == Approx(repeated.price).margin(1.0e-14));
        REQUIRE(first.standard_error == Approx(repeated.standard_error).margin(1.0e-14));
        REQUIRE(first.exercise_probabilities.size() == swaption.exercise_times.size());
        double probability = 0.0;
        for (double value : first.exercise_probabilities) {
          probability += value;
        }
        REQUIRE(probability == Approx(1.0).margin(1.0e-12));
      }
    }
  }
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

TEST_CASE("Curve and volatility scenarios are complete and directional") {
  const qf::InterpolatedYieldCurve curve({0.0, 1.0, 2.0, 5.0, 10.0},
                                         {1.0, 0.98, 0.955, 0.88, 0.75});
  const auto swap = qf::make_vanilla_swap(0.0, 10.0, 0.03, 1'000'000.0);
  const auto curve_scenarios = qf::run_swap_scenarios(swap, curve);
  REQUIRE(curve_scenarios.size() == 4U);
  REQUIRE(curve_scenarios[0].name == "parallel_up");
  REQUIRE(curve_scenarios[1].name == "parallel_down");
  REQUIRE(curve_scenarios[2].name == "steepener");
  REQUIRE(curve_scenarios[3].name == "flattener");

  const auto volatility_scenarios = qf::run_bachelier_volatility_scenarios(
      qf::OptionType::Call, 0.03, 0.03, 0.006, 2.0, 4.5, 0.001);
  REQUIRE(volatility_scenarios.size() == 2U);
  REQUIRE(volatility_scenarios[0].change > 0.0);
  REQUIRE(volatility_scenarios[1].change < 0.0);
}
