#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
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

struct G2ppMultiStartConfig {
  std::size_t starts{8};
  std::uint64_t seed{42U};
};

struct G2ppCalibrationRun {
  G2ppParameters initial;
  G2ppCalibrationResult calibration;
};

struct G2ppMultiStartResult {
  G2ppCalibrationResult best;
  std::size_t best_run{};
  std::size_t total_iterations{};
  std::vector<G2ppCalibrationRun> runs;
};

G2ppCalibrationResult calibrate_g2pp(YieldCurvePtr curve, std::span<const SwaptionQuote> quotes,
                                     G2ppParameters initial = {});

G2ppMultiStartResult calibrate_g2pp_multistart(YieldCurvePtr curve,
                                               std::span<const SwaptionQuote> quotes,
                                               G2ppMultiStartConfig config = {});

}  // namespace qf
