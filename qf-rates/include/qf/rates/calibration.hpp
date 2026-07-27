#pragma once

#include <cstddef>
#include <vector>

#include "qf/rates/g2pp.hpp"

namespace qf {

struct SwaptionQuote {
  Time expiry{};
  Time tenor{};
  Rate strike{};
  Volatility normal_volatility{};
  double weight{1.0};
};

struct CalibrationDiagnostic {
  Time expiry{};
  Time tenor{};
  Volatility market_volatility{};
  Volatility model_volatility{};
  double error_basis_points{};
};

struct G2ppCalibrationResult {
  G2ppParameters parameters;
  double rmse{};
  std::size_t iterations{};
  bool converged{};
  std::vector<CalibrationDiagnostic> diagnostics;
};

G2ppCalibrationResult calibrate_g2pp(YieldCurvePtr curve, std::span<const SwaptionQuote> quotes,
                                     G2ppParameters initial = {});

}  // namespace qf
