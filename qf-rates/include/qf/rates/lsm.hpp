#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qf/rates/g2pp.hpp"

namespace qf {

struct BermudanSwaption {
  std::vector<Time> exercise_times;
  Time maturity{};
  Rate strike{};
  Money notional{1.0};
  OptionType type{OptionType::Call};
  double fixed_frequency{1.0};
};

struct LsmConfig {
  std::size_t paths{20000};
  std::size_t steps_per_year{12};
  std::uint64_t seed{42U};
};

struct LsmResult {
  Money price{};
  double standard_error{};
  std::vector<double> exercise_probabilities;
};

LsmResult g2pp_bermudan_lsm(const G2ppModel& model, const BermudanSwaption& swaption,
                            LsmConfig config = {});

}  // namespace qf
