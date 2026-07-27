#include <chrono>
#include <iostream>
#include <memory>

#include "qf/rates/monte_carlo.hpp"

int main() {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const qf::EuropeanSwaption swaption{2.0, qf::Schedule(2.0, 12.0, 1.0), 0.026, 1'000'000.0};
  const auto start = std::chrono::steady_clock::now();
  constexpr std::size_t repetitions = 1000;
  double checksum = 0.0;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    checksum += qf::g2pp_european_swaption(model, swaption);
  }
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  std::cout << "quadrature_pricings=" << repetitions << " seconds=" << elapsed.count()
            << " prices_per_second=" << repetitions / elapsed.count() << " checksum=" << checksum
            << '\n';
}
