#pragma once

#include <memory>
#include <span>
#include <vector>

#include "qf/core/types.hpp"

namespace qf {

class YieldCurve {
 public:
  virtual ~YieldCurve() = default;
  [[nodiscard]] virtual DiscountFactor discount(Time maturity) const = 0;
  [[nodiscard]] Rate zero_rate(Time maturity) const;
  [[nodiscard]] Rate forward_rate(Time start, Time end) const;
};

class FlatYieldCurve final : public YieldCurve {
 public:
  explicit FlatYieldCurve(Rate continuously_compounded_rate);
  [[nodiscard]] DiscountFactor discount(Time maturity) const override;
  [[nodiscard]] Rate rate() const noexcept { return rate_; }

 private:
  Rate rate_{};
};

class InterpolatedYieldCurve final : public YieldCurve {
 public:
  InterpolatedYieldCurve(std::vector<Time> maturities, std::vector<DiscountFactor> discounts,
                         Interpolation interpolation = Interpolation::LogLinearDiscount);

  [[nodiscard]] DiscountFactor discount(Time maturity) const override;
  [[nodiscard]] const std::vector<Time>& maturities() const noexcept { return maturities_; }
  [[nodiscard]] const std::vector<DiscountFactor>& discounts() const noexcept { return discounts_; }
  [[nodiscard]] Interpolation interpolation() const noexcept { return interpolation_; }
  [[nodiscard]] InterpolatedYieldCurve bumped(double bump, std::size_t bucket) const;
  [[nodiscard]] InterpolatedYieldCurve parallel_bumped(double bump) const;

 private:
  std::vector<Time> maturities_;
  std::vector<DiscountFactor> discounts_;
  Interpolation interpolation_;
};

using YieldCurvePtr = std::shared_ptr<const YieldCurve>;

}  // namespace qf
