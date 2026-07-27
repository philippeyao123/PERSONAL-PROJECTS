#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "qf/rates/calibration.hpp"
#include "qf/rates/lsm.hpp"
#include "qf/rates/monte_carlo.hpp"
#include "qf/rates/risk.hpp"
#include "qf/rates/xva.hpp"

int main() {
  try {
    auto curve = std::make_shared<qf::InterpolatedYieldCurve>(
        std::vector<double>{0, 1, 2, 5, 10, 20},
        std::vector<double>{1, .976, .949, .875, .755, .56});
    const auto base_swap = qf::make_vanilla_swap(2, 7, .03, 1'000'000);
    const double strike = base_swap.par_rate(*curve, *curve);
    const std::vector<qf::SwaptionQuote> quotes{
        {1, 5, strike, .0055, 1}, {2, 5, strike, .0062, 1}, {5, 5, strike, .0068, 1}};
    const auto calibration = qf::calibrate_g2pp(curve, quotes);
    const qf::G2ppModel model(curve, calibration.parameters);
    const qf::EuropeanSwaption european{2, qf::Schedule(2, 7, 1), strike, 1'000'000};
    const double european_price = qf::g2pp_european_swaption(model, european);
    const auto mc = qf::g2pp_european_swaption_mc(model, european, {12000, 120, 42, true});
    const auto bermudan =
        qf::g2pp_bermudan_lsm(model, {{1, 2, 3}, 7, strike, 1'000'000}, {8000, 12, 42});
    const auto dv01 = qf::swap_dv01(base_swap, *curve);
    const auto exposure =
        qf::simulate_swap_exposure(model, std::vector<qf::InterestRateSwap>{base_swap},
                                   std::vector<double>{1, 2, 3, 4, 5, 6}, 3000);
    const auto xva = qf::compute_xva(*curve, exposure);
    std::cout << std::fixed << std::setprecision(2)
              << "G2++ calibration RMSE: " << calibration.rmse * 10000 << " bp\n"
              << "European: " << european_price << "\n"
              << "Monte-Carlo: " << mc.price << " +/- " << 1.96 * mc.standard_error << "\n"
              << "Bermudan LSM: " << bermudan.price << "\n"
              << "Swap DV01: " << dv01.parallel_dv01 << "\n"
              << "CVA/DVA/FVA: " << xva.cva << " / " << xva.dva << " / " << xva.fva << '\n';
  } catch (const std::exception& error) {
    std::cerr << "qf-rates error: " << error.what() << '\n';
    return 1;
  }
}
