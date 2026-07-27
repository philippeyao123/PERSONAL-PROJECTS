#pragma once

#include "qf/core/types.hpp"
#include "qf/rates/yield_curve.hpp"

namespace qf {

struct FixedLeg {
  Schedule schedule;
  Rate coupon{};
  Money notional{1.0};
  PayReceive direction{PayReceive::Pay};
};

struct FloatingLeg {
  Schedule schedule;
  Rate spread{};
  Money notional{1.0};
  PayReceive direction{PayReceive::Receive};
};

class InterestRateSwap {
 public:
  InterestRateSwap(FixedLeg fixed_leg, FloatingLeg floating_leg);

  [[nodiscard]] const FixedLeg& fixed_leg() const noexcept { return fixed_leg_; }
  [[nodiscard]] const FloatingLeg& floating_leg() const noexcept { return floating_leg_; }
  [[nodiscard]] Money fixed_leg_npv(const YieldCurve& discount_curve) const;
  [[nodiscard]] Money floating_leg_npv(const YieldCurve& discount_curve,
                                       const YieldCurve& forward_curve) const;
  [[nodiscard]] Money npv(const YieldCurve& discount_curve, const YieldCurve& forward_curve) const;
  [[nodiscard]] Rate par_rate(const YieldCurve& discount_curve,
                              const YieldCurve& forward_curve) const;
  [[nodiscard]] double annuity(const YieldCurve& discount_curve) const;

 private:
  FixedLeg fixed_leg_;
  FloatingLeg floating_leg_;
};

InterestRateSwap make_vanilla_swap(Time start, Time maturity, Rate fixed_rate, Money notional = 1.0,
                                   PayReceive fixed_direction = PayReceive::Pay,
                                   double fixed_frequency = 1.0, double floating_frequency = 0.5,
                                   Rate spread = 0.0);

}  // namespace qf
