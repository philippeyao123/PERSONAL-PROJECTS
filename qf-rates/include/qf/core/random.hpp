#pragma once

#include <cstdint>
#include <random>
#include <utility>

namespace qf {

class RandomEngine {
 public:
  explicit RandomEngine(std::uint64_t seed = 42U);

  [[nodiscard]] double uniform();
  [[nodiscard]] double normal();
  [[nodiscard]] std::pair<double, double> antithetic_normal();
  void reseed(std::uint64_t seed);

 private:
  std::mt19937_64 engine_;
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
  std::normal_distribution<double> normal_{0.0, 1.0};
};

}  // namespace qf
