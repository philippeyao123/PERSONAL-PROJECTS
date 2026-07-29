#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>

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

TEST_CASE("Multi-start calibration is reproducible and dominates the default start") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const std::vector<qf::SwaptionQuote> quotes{
      {1.0, 5.0, 0.026, 0.0060, 1.0},
      {2.0, 5.0, 0.026, 0.0065, 1.0},
      {3.0, 10.0, 0.026, 0.0072, 1.0},
  };
  const qf::G2ppMultiStartConfig config{3, 2026};
  const auto first = qf::calibrate_g2pp_multistart(curve, quotes, config);
  const auto repeated = qf::calibrate_g2pp_multistart(curve, quotes, config);
  REQUIRE(first.runs.size() == config.starts);
  REQUIRE(first.best_run < first.runs.size());
  REQUIRE(first.best.rmse <= first.runs.front().calibration.rmse + 1.0e-15);
  REQUIRE(first.best.rmse == Approx(repeated.best.rmse).margin(1.0e-14));
  REQUIRE(first.best_run == repeated.best_run);
  REQUIRE(first.total_iterations > 0U);
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

TEST_CASE("Out-of-sample LSM evaluates a frozen policy on independent paths") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const qf::BermudanSwaption bermudan{{1.0, 2.0, 3.0}, 6.0, 0.026, 1.0, qf::OptionType::Call, 1.0};
  const qf::LsmOutOfSampleConfig config{3000, 5000, 8, 71, 1907, qf::LsmBasis::Quadratic};
  const auto first = qf::g2pp_bermudan_lsm_out_of_sample(model, bermudan, config);
  const auto repeated = qf::g2pp_bermudan_lsm_out_of_sample(model, bermudan, config);
  REQUIRE(first.price >= 0.0);
  REQUIRE(first.training_price >= 0.0);
  REQUIRE(first.standard_error > 0.0);
  REQUIRE(first.price == Approx(repeated.price).margin(1.0e-14));
  REQUIRE(first.training_price == Approx(repeated.training_price).margin(1.0e-14));
  double probability = first.non_exercise_probability;
  for (double exercise_probability : first.exercise_probabilities) {
    probability += exercise_probability;
  }
  REQUIRE(probability == Approx(1.0).margin(1.0e-12));
}

TEST_CASE("G2++ stress grid is finite monotone in strike and increasing in volatility") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppParameters low_volatility{0.10, 0.30, 0.005, 0.0075, -0.70};
  const qf::G2ppParameters high_volatility{0.10, 0.30, 0.020, 0.0300, -0.70};
  for (const double expiry : {1.0, 2.0, 5.0}) {
    for (const double tenor : {2.0, 5.0, 10.0}) {
      const auto swap = qf::make_vanilla_swap(expiry, expiry + tenor, 0.025);
      const double forward = swap.par_rate(*curve, *curve);
      double previous = std::numeric_limits<double>::infinity();
      for (const double moneyness_basis_points : {-100.0, 0.0, 100.0}) {
        const double strike = forward + moneyness_basis_points * 1.0e-4;
        const qf::EuropeanSwaption swaption{expiry, qf::Schedule(expiry, expiry + tenor, 1.0),
                                            strike, 1.0, qf::OptionType::Call};
        const double low =
            qf::g2pp_european_swaption(qf::G2ppModel(curve, low_volatility), swaption);
        const double high =
            qf::g2pp_european_swaption(qf::G2ppModel(curve, high_volatility), swaption);
        REQUIRE(std::isfinite(low));
        REQUIRE(std::isfinite(high));
        REQUIRE(low <= previous + 1.0e-14);
        REQUIRE(high >= low - 1.0e-14);
        previous = low;
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
