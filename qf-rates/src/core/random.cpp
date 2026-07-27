#include "qf/core/random.hpp"

namespace qf {

RandomEngine::RandomEngine(std::uint64_t seed) : engine_(seed) {}

double RandomEngine::uniform() { return uniform_(engine_); }

double RandomEngine::normal() { return normal_(engine_); }

std::pair<double, double> RandomEngine::antithetic_normal() {
  const double draw = normal();
  return {draw, -draw};
}

void RandomEngine::reseed(std::uint64_t seed) {
  engine_.seed(seed);
  normal_.reset();
}

}  // namespace qf
