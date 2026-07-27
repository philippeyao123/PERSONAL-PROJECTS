#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "qf/core/error.hpp"

namespace qf {

using Rate = double;
using Volatility = double;
using Time = double;
using DiscountFactor = double;
using Money = double;

enum class OptionType { Call, Put };
enum class PayReceive { Pay, Receive };
enum class Interpolation { LinearDiscount, LogLinearDiscount };

struct Cashflow {
  Time payment_time{};
  Money amount{};
};

struct CouponPeriod {
  Time start{};
  Time end{};
  Time payment{};
  double accrual{};
};

class Schedule {
 public:
  Schedule(Time start, Time end, double frequency);

  [[nodiscard]] const std::vector<CouponPeriod>& periods() const noexcept { return periods_; }
  [[nodiscard]] Time start() const noexcept { return start_; }
  [[nodiscard]] Time end() const noexcept { return end_; }

 private:
  Time start_{};
  Time end_{};
  std::vector<CouponPeriod> periods_;
};

inline Schedule::Schedule(Time start, Time end, double frequency) : start_(start), end_(end) {
  if (!(start >= 0.0 && end > start && frequency > 0.0)) {
    throw ValidationError("Schedule requires 0 <= start < end and frequency > 0");
  }
  constexpr double tolerance = 1.0e-12;
  Time current = start;
  while (current < end - tolerance) {
    const Time next = std::min(current + frequency, end);
    periods_.push_back({current, next, next, next - current});
    current = next;
    if (periods_.size() > 10000U) {
      throw ValidationError("Schedule contains too many periods");
    }
  }
}

}  // namespace qf
