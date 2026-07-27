#include "qf/rates/yield_curve.hpp"

#include <algorithm>
#include <cmath>

#include "qf/core/error.hpp"

namespace qf {

Rate YieldCurve::zero_rate(Time maturity) const {
  if (maturity < 0.0) {
    throw ValidationError("Maturity cannot be negative");
  }
  if (maturity == 0.0) {
    constexpr double epsilon = 1.0e-8;
    return -std::log(discount(epsilon)) / epsilon;
  }
  return -std::log(discount(maturity)) / maturity;
}

Rate YieldCurve::forward_rate(Time start, Time end) const {
  if (!(start >= 0.0 && end > start)) {
    throw ValidationError("Forward interval requires 0 <= start < end");
  }
  return std::log(discount(start) / discount(end)) / (end - start);
}

FlatYieldCurve::FlatYieldCurve(Rate continuously_compounded_rate)
    : rate_(continuously_compounded_rate) {
  if (!std::isfinite(rate_)) {
    throw ValidationError("Flat curve rate must be finite");
  }
}

DiscountFactor FlatYieldCurve::discount(Time maturity) const {
  if (maturity < 0.0) {
    throw ValidationError("Maturity cannot be negative");
  }
  return std::exp(-rate_ * maturity);
}

InterpolatedYieldCurve::InterpolatedYieldCurve(std::vector<Time> maturities,
                                               std::vector<DiscountFactor> discounts,
                                               Interpolation interpolation)
    : maturities_(std::move(maturities)),
      discounts_(std::move(discounts)),
      interpolation_(interpolation) {
  if (maturities_.empty() || maturities_.size() != discounts_.size()) {
    throw ValidationError("Curve nodes must have equal non-zero sizes");
  }
  if (maturities_.front() != 0.0) {
    maturities_.insert(maturities_.begin(), 0.0);
    discounts_.insert(discounts_.begin(), 1.0);
  }
  for (std::size_t index = 0; index < maturities_.size(); ++index) {
    if (maturities_[index] < 0.0 || !(discounts_[index] > 0.0) ||
        !std::isfinite(discounts_[index])) {
      throw ValidationError("Curve maturities and discounts must be valid and positive");
    }
    if (index > 0U && !(maturities_[index] > maturities_[index - 1U])) {
      throw ValidationError("Curve maturities must be strictly increasing");
    }
  }
  if (std::abs(discounts_.front() - 1.0) > 1.0e-12) {
    throw ValidationError("Discount factor at time zero must equal one");
  }
}

DiscountFactor InterpolatedYieldCurve::discount(Time maturity) const {
  if (maturity < 0.0) {
    throw ValidationError("Maturity cannot be negative");
  }
  if (maturity == 0.0) {
    return 1.0;
  }
  const auto upper = std::upper_bound(maturities_.begin(), maturities_.end(), maturity);
  if (upper == maturities_.begin()) {
    return discounts_.front();
  }
  if (upper == maturities_.end()) {
    const std::size_t last = maturities_.size() - 1U;
    const Rate terminal_rate = -std::log(discounts_[last]) / maturities_[last];
    return std::exp(-terminal_rate * maturity);
  }
  const std::size_t right = static_cast<std::size_t>(upper - maturities_.begin());
  const std::size_t left = right - 1U;
  const double weight = (maturity - maturities_[left]) / (maturities_[right] - maturities_[left]);
  if (interpolation_ == Interpolation::LinearDiscount) {
    return discounts_[left] + weight * (discounts_[right] - discounts_[left]);
  }
  return std::exp(std::log(discounts_[left]) +
                  weight * (std::log(discounts_[right]) - std::log(discounts_[left])));
}

InterpolatedYieldCurve InterpolatedYieldCurve::bumped(double bump, std::size_t bucket) const {
  if (bucket >= maturities_.size() || bucket == 0U) {
    throw ValidationError("Curve bump bucket must refer to a non-zero maturity node");
  }
  auto bumped_discounts = discounts_;
  bumped_discounts[bucket] *= std::exp(-bump * maturities_[bucket]);
  return {maturities_, std::move(bumped_discounts), interpolation_};
}

InterpolatedYieldCurve InterpolatedYieldCurve::parallel_bumped(double bump) const {
  auto bumped_discounts = discounts_;
  for (std::size_t index = 1; index < maturities_.size(); ++index) {
    bumped_discounts[index] *= std::exp(-bump * maturities_[index]);
  }
  return {maturities_, std::move(bumped_discounts), interpolation_};
}

}  // namespace qf
