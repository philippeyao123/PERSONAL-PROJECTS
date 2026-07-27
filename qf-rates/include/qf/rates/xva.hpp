#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "qf/rates/g2pp.hpp"
#include "qf/rates/swap.hpp"

namespace qf {

struct ExposureProfile {
  std::vector<Time> times;
  std::vector<Money> epe;
  std::vector<Money> ene;
};

struct XvaAssumptions {
  double counterparty_hazard_rate{0.02};
  double own_hazard_rate{0.015};
  double recovery_rate{0.40};
  double funding_spread{0.01};
};

struct XvaResult {
  Money cva{};
  Money dva{};
  Money fva{};
  ExposureProfile profile;
};

ExposureProfile simulate_swap_exposure(const G2ppModel& model,
                                       std::span<const InterestRateSwap> swaps,
                                       std::span<const Time> exposure_times,
                                       std::size_t paths = 10000, std::uint64_t seed = 42U,
                                       bool netting = true, double wrong_way_beta = 0.0);

XvaResult compute_xva(const YieldCurve& curve, ExposureProfile profile,
                      XvaAssumptions assumptions = {});

Money simplified_simm(double absolute_dv01, double absolute_vega, double rates_delta_weight = 50.0,
                      double rates_vega_weight = 0.21);

Money compute_mva(Money initial_margin, double funding_spread, Time horizon,
                  DiscountFactor average_discount = 1.0);

}  // namespace qf
