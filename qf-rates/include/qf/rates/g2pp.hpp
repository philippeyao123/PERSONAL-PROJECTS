#pragma once

#include <array>
#include <memory>

#include "qf/core/types.hpp"
#include "qf/rates/yield_curve.hpp"

namespace qf {

struct G2ppParameters {
  double a{0.10};
  double b{0.30};
  double sigma{0.01};
  double eta{0.015};
  double rho{-0.70};
};

struct FactorState {
  double x{};
  double y{};
};

struct EuropeanSwaption {
  Time expiry{};
  Schedule underlying_schedule;
  Rate strike{};
  Money notional{1.0};
  OptionType type{OptionType::Call};  // Call = payer, Put = receiver.
};

class G2ppModel {
 public:
  G2ppModel(YieldCurvePtr curve, G2ppParameters parameters);

  [[nodiscard]] const YieldCurve& curve() const noexcept { return *curve_; }
  [[nodiscard]] const YieldCurvePtr& curve_ptr() const noexcept { return curve_; }
  [[nodiscard]] const G2ppParameters& parameters() const noexcept { return parameters_; }
  [[nodiscard]] double factor_loading(double mean_reversion, Time horizon) const;
  [[nodiscard]] double integrated_variance(Time horizon) const;
  [[nodiscard]] std::array<std::array<double, 2>, 2> factor_covariance(Time horizon) const;
  [[nodiscard]] DiscountFactor discount_bond(Time observation, Time maturity,
                                             FactorState state = {}) const;
  [[nodiscard]] Rate short_rate(Time observation, FactorState state) const;
  [[nodiscard]] FactorState evolve(FactorState state, Time step, double normal_x,
                                   double normal_independent) const;

 private:
  YieldCurvePtr curve_;
  G2ppParameters parameters_;
};

Money g2pp_european_swaption(const G2ppModel& model, const EuropeanSwaption& swaption,
                             std::size_t quadrature_order = 8);

}  // namespace qf
