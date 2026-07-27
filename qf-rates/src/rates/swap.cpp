#include "qf/rates/swap.hpp"

#include <cmath>
#include <utility>

#include "qf/core/error.hpp"

namespace qf {
namespace {

double sign(PayReceive direction) { return direction == PayReceive::Receive ? 1.0 : -1.0; }

}  // namespace

InterestRateSwap::InterestRateSwap(FixedLeg fixed_leg, FloatingLeg floating_leg)
    : fixed_leg_(std::move(fixed_leg)), floating_leg_(std::move(floating_leg)) {
  if (!(fixed_leg_.notional > 0.0 && floating_leg_.notional > 0.0)) {
    throw ValidationError("Swap notionals must be positive");
  }
  if (std::abs(fixed_leg_.notional - floating_leg_.notional) > 1.0e-12) {
    throw ValidationError("Fixed and floating leg notionals must match");
  }
  if (fixed_leg_.direction == floating_leg_.direction) {
    throw ValidationError("Swap legs must have opposite directions");
  }
}

double InterestRateSwap::annuity(const YieldCurve& discount_curve) const {
  double value = 0.0;
  for (const auto& period : fixed_leg_.schedule.periods()) {
    value += period.accrual * discount_curve.discount(period.payment);
  }
  return fixed_leg_.notional * value;
}

Money InterestRateSwap::fixed_leg_npv(const YieldCurve& discount_curve) const {
  return sign(fixed_leg_.direction) * fixed_leg_.coupon * annuity(discount_curve);
}

Money InterestRateSwap::floating_leg_npv(const YieldCurve& discount_curve,
                                         const YieldCurve& forward_curve) const {
  double value = 0.0;
  for (const auto& period : floating_leg_.schedule.periods()) {
    const double simple_forward =
        (forward_curve.discount(period.start) / forward_curve.discount(period.end) - 1.0) /
        period.accrual;
    value += period.accrual * (simple_forward + floating_leg_.spread) *
             discount_curve.discount(period.payment);
  }
  return sign(floating_leg_.direction) * floating_leg_.notional * value;
}

Money InterestRateSwap::npv(const YieldCurve& discount_curve,
                            const YieldCurve& forward_curve) const {
  return fixed_leg_npv(discount_curve) + floating_leg_npv(discount_curve, forward_curve);
}

Rate InterestRateSwap::par_rate(const YieldCurve& discount_curve,
                                const YieldCurve& forward_curve) const {
  double floating_unsigned = 0.0;
  for (const auto& period : floating_leg_.schedule.periods()) {
    const double simple_forward =
        (forward_curve.discount(period.start) / forward_curve.discount(period.end) - 1.0) /
        period.accrual;
    floating_unsigned += period.accrual * (simple_forward + floating_leg_.spread) *
                         discount_curve.discount(period.payment);
  }
  const double unit_annuity = annuity(discount_curve) / fixed_leg_.notional;
  if (unit_annuity <= 0.0) {
    throw NumericalError("Swap annuity must be positive");
  }
  return floating_unsigned / unit_annuity;
}

InterestRateSwap make_vanilla_swap(Time start, Time maturity, Rate fixed_rate, Money notional,
                                   PayReceive fixed_direction, double fixed_frequency,
                                   double floating_frequency, Rate spread) {
  const PayReceive floating_direction =
      fixed_direction == PayReceive::Pay ? PayReceive::Receive : PayReceive::Pay;
  return {{Schedule(start, maturity, fixed_frequency), fixed_rate, notional, fixed_direction},
          {Schedule(start, maturity, floating_frequency), spread, notional, floating_direction}};
}

}  // namespace qf
