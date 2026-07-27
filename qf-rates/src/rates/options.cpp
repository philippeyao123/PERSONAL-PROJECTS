#include "qf/rates/options.hpp"

#include <cmath>

#include "qf/core/error.hpp"
#include "qf/core/numerics.hpp"

namespace qf {
namespace {

double option_sign(OptionType type) { return type == OptionType::Call ? 1.0 : -1.0; }

void validate_scale(double volatility, Time expiry, DiscountFactor discount, Money notional) {
  if (!(volatility >= 0.0 && expiry >= 0.0 && discount > 0.0 && notional >= 0.0)) {
    throw ValidationError(
        "Volatility, expiry and notional must be non-negative and discount must be positive");
  }
}

}  // namespace

OptionResult black76(OptionType type, double forward, double strike, Volatility volatility,
                     Time expiry, DiscountFactor discount, Money notional) {
  validate_scale(volatility, expiry, discount, notional);
  if (!(forward > 0.0 && strike >= 0.0)) {
    throw ValidationError("Black-76 requires a positive forward and non-negative strike");
  }
  const double sign = option_sign(type);
  const double scale = discount * notional;
  const double standard_deviation = volatility * std::sqrt(expiry);
  if (standard_deviation < 1.0e-14 || strike == 0.0) {
    const double intrinsic = std::max(sign * (forward - strike), 0.0);
    const double delta = sign * (sign * (forward - strike) > 0.0 ? 1.0 : 0.0) * scale;
    return {scale * intrinsic, delta, 0.0, 0.0};
  }
  const double d1 = (std::log(forward / strike) + 0.5 * standard_deviation * standard_deviation) /
                    standard_deviation;
  const double d2 = d1 - standard_deviation;
  const double price =
      scale * sign * (forward * normal_cdf(sign * d1) - strike * normal_cdf(sign * d2));
  const double delta = scale * sign * normal_cdf(sign * d1);
  const double gamma = scale * normal_pdf(d1) / (forward * standard_deviation);
  const double vega = scale * forward * normal_pdf(d1) * std::sqrt(expiry);
  return {price, delta, gamma, vega};
}

OptionResult bachelier(OptionType type, double forward, double strike, Volatility normal_volatility,
                       Time expiry, DiscountFactor discount, Money notional) {
  validate_scale(normal_volatility, expiry, discount, notional);
  if (!std::isfinite(forward) || !std::isfinite(strike)) {
    throw ValidationError("Bachelier forward and strike must be finite");
  }
  const double sign = option_sign(type);
  const double scale = discount * notional;
  const double standard_deviation = normal_volatility * std::sqrt(expiry);
  if (standard_deviation < 1.0e-14) {
    const double intrinsic = std::max(sign * (forward - strike), 0.0);
    const double delta = sign * (sign * (forward - strike) > 0.0 ? 1.0 : 0.0) * scale;
    return {scale * intrinsic, delta, 0.0, 0.0};
  }
  const double d = (forward - strike) / standard_deviation;
  const double price = scale * (sign * (forward - strike) * normal_cdf(sign * d) +
                                standard_deviation * normal_pdf(d));
  const double delta = scale * sign * normal_cdf(sign * d);
  const double gamma = scale * normal_pdf(d) / standard_deviation;
  const double vega = scale * std::sqrt(expiry) * normal_pdf(d);
  return {price, delta, gamma, vega};
}

Money black76_swaption(OptionType type, Rate forward_swap_rate, Rate strike, Volatility volatility,
                       Time expiry, double annuity, Money notional) {
  if (!(annuity > 0.0)) {
    throw ValidationError("Swaption annuity must be positive");
  }
  return black76(type, forward_swap_rate, strike, volatility, expiry, 1.0, annuity * notional)
      .price;
}

Money bachelier_swaption(OptionType type, Rate forward_swap_rate, Rate strike,
                         Volatility normal_volatility, Time expiry, double annuity,
                         Money notional) {
  if (!(annuity > 0.0)) {
    throw ValidationError("Swaption annuity must be positive");
  }
  return bachelier(type, forward_swap_rate, strike, normal_volatility, expiry, 1.0,
                   annuity * notional)
      .price;
}

}  // namespace qf
