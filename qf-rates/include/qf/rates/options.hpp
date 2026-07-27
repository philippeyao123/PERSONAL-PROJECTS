#pragma once

#include "qf/core/types.hpp"

namespace qf {

struct OptionResult {
  Money price{};
  double delta{};
  double gamma{};
  double vega{};
};

OptionResult black76(OptionType type, double forward, double strike, Volatility volatility,
                     Time expiry, DiscountFactor discount = 1.0, Money notional = 1.0);

OptionResult bachelier(OptionType type, double forward, double strike, Volatility normal_volatility,
                       Time expiry, DiscountFactor discount = 1.0, Money notional = 1.0);

Money black76_swaption(OptionType type, Rate forward_swap_rate, Rate strike, Volatility volatility,
                       Time expiry, double annuity, Money notional = 1.0);

Money bachelier_swaption(OptionType type, Rate forward_swap_rate, Rate strike,
                         Volatility normal_volatility, Time expiry, double annuity,
                         Money notional = 1.0);

}  // namespace qf
