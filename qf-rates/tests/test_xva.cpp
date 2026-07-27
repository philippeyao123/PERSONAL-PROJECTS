#include <catch2/catch_test_macros.hpp>

#include "qf/rates/xva.hpp"

TEST_CASE("Netting does not increase expected positive exposure") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const std::vector<qf::InterestRateSwap> swaps{
      qf::make_vanilla_swap(0.0, 5.0, 0.02, 1'000'000.0),
      qf::make_vanilla_swap(0.0, 5.0, 0.03, 1'000'000.0, qf::PayReceive::Receive)};
  const std::vector<double> times{0.5, 1.0, 2.0, 3.0, 4.0};
  const auto gross = qf::simulate_swap_exposure(model, swaps, times, 1000, 7, false);
  const auto net = qf::simulate_swap_exposure(model, swaps, times, 1000, 7, true);
  for (std::size_t index = 0; index < times.size(); ++index) {
    REQUIRE(net.epe[index] <= gross.epe[index] + 1.0e-10);
  }
}

TEST_CASE("CVA DVA FVA SIMM and MVA are non-negative") {
  const qf::FlatYieldCurve curve(0.02);
  qf::ExposureProfile profile{
      {1.0, 2.0, 3.0}, {100000.0, 80000.0, 50000.0}, {40000.0, 30000.0, 20000.0}};
  const auto result = qf::compute_xva(curve, profile);
  REQUIRE(result.cva >= 0.0);
  REQUIRE(result.dva >= 0.0);
  REQUIRE(result.fva >= 0.0);
  const double margin = qf::simplified_simm(100.0, 5000.0);
  REQUIRE(margin > 0.0);
  REQUIRE(qf::compute_mva(margin, 0.01, 3.0, 0.95) >= 0.0);
}
