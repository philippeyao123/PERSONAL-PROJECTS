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

enum class LsmBasis { Linear, Quadratic };

struct LsmConfig {
  std::size_t paths{20000};
  std::size_t steps_per_year{12};
  std::uint64_t seed{42U};
  LsmBasis basis{LsmBasis::Quadratic};
};

struct LsmResult {
  Money price{};
  double standard_error{};
  std::vector<double> exercise_probabilities;
};

struct LsmOutOfSampleConfig {
  std::size_t training_paths{20000};
  std::size_t valuation_paths{50000};
  std::size_t steps_per_year{12};
  std::uint64_t training_seed{42U};
  std::uint64_t valuation_seed{314159U};
  LsmBasis basis{LsmBasis::Quadratic};
};

struct LsmOutOfSampleResult {
  Money training_price{};
  Money price{};
  double standard_error{};
  std::vector<double> exercise_probabilities;
  double non_exercise_probability{};
};

LsmResult g2pp_bermudan_lsm(const G2ppModel& model, const BermudanSwaption& swaption,
                            LsmConfig config = {});

LsmOutOfSampleResult g2pp_bermudan_lsm_out_of_sample(const G2ppModel& model,
                                                     const BermudanSwaption& swaption,
                                                     LsmOutOfSampleConfig config = {});

}  // namespace qf
