#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "qf/rates/lsm.hpp"
#include "qf/rates/monte_carlo.hpp"
#include "qf/rates/xva.hpp"

int main() {
  constexpr double forward = 100.0;
  constexpr double strike = 100.0;
  constexpr double volatility = 0.20;
  const auto terminal = [](double normal) {
    return forward * std::exp(-0.5 * volatility * volatility + volatility * normal);
  };
  const auto payoff = [&](double normal) { return std::max(terminal(normal) - strike, 0.0); };
  const auto plain = qf::monte_carlo_normal(payoff, {100000, 1, 17, false});
  const auto antithetic = qf::monte_carlo_normal(payoff, {100000, 1, 17, true});
  const auto controlled =
      qf::monte_carlo_normal(payoff, {100000, 1, 17, false}, terminal, forward);

  std::cout << std::fixed << std::setprecision(8);
  std::cout << "# Variance reduction\n\n"
            << "| Method | Standard error | Variance ratio vs plain |\n"
            << "|---|---:|---:|\n";
  const auto row = [&](const char* name, const qf::MonteCarloResult& result) {
    const double ratio =
        std::pow(result.standard_error / plain.standard_error, 2.0);
    std::cout << "| " << name << " | " << result.standard_error << " | " << ratio << " |\n";
  };
  row("Plain", plain);
  row("Antithetic", antithetic);
  row("Control variate", controlled);

  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const qf::BermudanSwaption bermudan{{1.0, 2.0, 3.0}, 6.0, 0.026, 1.0,
                                       qf::OptionType::Call, 1.0};
  std::cout << "\n# LSM convergence\n\n"
            << "| Paths | Seed | Basis | Price | Standard error |\n"
            << "|---:|---:|---|---:|---:|\n";
  for (const std::size_t paths : {2000U, 5000U, 10000U}) {
    for (const std::uint64_t seed : {7U, 42U}) {
      for (const auto basis : {qf::LsmBasis::Linear, qf::LsmBasis::Quadratic}) {
        const auto result =
            qf::g2pp_bermudan_lsm(model, bermudan, {paths, 12, seed, basis});
        std::cout << "| " << paths << " | " << seed << " | "
                  << (basis == qf::LsmBasis::Linear ? "Linear" : "Quadratic") << " | "
                  << result.price << " | " << result.standard_error << " |\n";
      }
    }
  }

  const std::vector<qf::InterestRateSwap> swaps{
      qf::make_vanilla_swap(0.0, 7.0, 0.02, 1'000'000.0)};
  const std::vector<double> times{0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
  const auto independent =
      qf::simulate_swap_exposure(model, swaps, times, 10000, 19, true, 0.0);
  const auto wrong_way =
      qf::simulate_swap_exposure(model, swaps, times, 10000, 19, true, 20.0);
  const auto independent_xva = qf::compute_xva(*curve, independent);
  const auto wrong_way_xva = qf::compute_xva(*curve, wrong_way);
  std::cout << "\n# Wrong-way risk\n\n"
            << "| Case | CVA |\n"
            << "|---|---:|\n"
            << "| Independent exposure | " << independent_xva.cva << " |\n"
            << "| Proxy WWR beta=20 | " << wrong_way_xva.cva << " |\n";
}
