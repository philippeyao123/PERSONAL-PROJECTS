#include "qf/rates/xva.hpp"

#include <algorithm>
#include <cmath>

#include "qf/core/error.hpp"
#include "qf/core/random.hpp"

namespace qf {
namespace {

double swap_mark_to_market(const G2ppModel& model, const InterestRateSwap& swap, Time time,
                           FactorState state) {
  double fixed = 0.0;
  for (const auto& period : swap.fixed_leg().schedule.periods()) {
    if (period.payment > time) {
      fixed += period.accrual * model.discount_bond(time, period.payment, state);
    }
  }
  fixed *= swap.fixed_leg().coupon * swap.fixed_leg().notional;
  const double fixed_sign = swap.fixed_leg().direction == PayReceive::Receive ? 1.0 : -1.0;
  const double floating_sign = -fixed_sign;
  const auto& floating_periods = swap.floating_leg().schedule.periods();
  double floating = 0.0;
  for (const auto& period : floating_periods) {
    if (period.payment <= time) {
      continue;
    }
    const Time effective_start = std::max(time, period.start);
    const double forward = (model.discount_bond(time, effective_start, state) /
                                model.discount_bond(time, period.end, state) -
                            1.0) /
                           period.accrual;
    floating += period.accrual * (forward + swap.floating_leg().spread) *
                model.discount_bond(time, period.payment, state);
  }
  floating *= swap.floating_leg().notional;
  return fixed_sign * fixed + floating_sign * floating;
}

}  // namespace

ExposureProfile simulate_swap_exposure(const G2ppModel& model,
                                       std::span<const InterestRateSwap> swaps,
                                       std::span<const Time> exposure_times, std::size_t paths,
                                       std::uint64_t seed, bool netting, double wrong_way_beta) {
  if (swaps.empty() || exposure_times.empty() || paths == 0U || wrong_way_beta < 0.0) {
    throw ValidationError("Invalid exposure simulation inputs");
  }
  for (std::size_t index = 0; index < exposure_times.size(); ++index) {
    if (exposure_times[index] <= 0.0 ||
        (index > 0U && exposure_times[index] <= exposure_times[index - 1U])) {
      throw ValidationError("Exposure times must be positive and strictly increasing");
    }
  }
  ExposureProfile profile;
  profile.times.assign(exposure_times.begin(), exposure_times.end());
  profile.epe.assign(exposure_times.size(), 0.0);
  profile.ene.assign(exposure_times.size(), 0.0);
  RandomEngine random(seed);
  for (std::size_t path = 0; path < paths; ++path) {
    FactorState state{};
    Time previous = 0.0;
    for (std::size_t date = 0; date < exposure_times.size(); ++date) {
      const double step = exposure_times[date] - previous;
      state = model.evolve(state, step, random.normal(), random.normal());
      previous = exposure_times[date];
      double positive = 0.0;
      double negative = 0.0;
      double net_value = 0.0;
      for (const auto& swap : swaps) {
        const double value = swap_mark_to_market(model, swap, previous, state);
        net_value += value;
        positive += std::max(value, 0.0);
        negative += std::max(-value, 0.0);
      }
      if (netting) {
        positive = std::max(net_value, 0.0);
        negative = std::max(-net_value, 0.0);
      }
      const double wrong_way_multiplier = std::exp(wrong_way_beta * (state.x + state.y));
      profile.epe[date] += positive * wrong_way_multiplier / static_cast<double>(paths);
      profile.ene[date] += negative / static_cast<double>(paths);
    }
  }
  return profile;
}

XvaResult compute_xva(const YieldCurve& curve, ExposureProfile profile,
                      XvaAssumptions assumptions) {
  if (profile.times.empty() || profile.times.size() != profile.epe.size() ||
      profile.times.size() != profile.ene.size() || assumptions.counterparty_hazard_rate < 0.0 ||
      assumptions.own_hazard_rate < 0.0 || assumptions.recovery_rate < 0.0 ||
      assumptions.recovery_rate > 1.0 || assumptions.funding_spread < 0.0) {
    throw ValidationError("Invalid XVA profile or assumptions");
  }
  double cva = 0.0;
  double dva = 0.0;
  double fva = 0.0;
  Time previous = 0.0;
  double counterparty_survival = 1.0;
  double own_survival = 1.0;
  for (std::size_t index = 0; index < profile.times.size(); ++index) {
    const double step = profile.times[index] - previous;
    if (!(step > 0.0)) {
      throw ValidationError("XVA times must be strictly increasing");
    }
    const double next_counterparty_survival =
        std::exp(-assumptions.counterparty_hazard_rate * profile.times[index]);
    const double next_own_survival = std::exp(-assumptions.own_hazard_rate * profile.times[index]);
    const double discount = curve.discount(profile.times[index]);
    cva += (1.0 - assumptions.recovery_rate) * discount * profile.epe[index] *
           (counterparty_survival - next_counterparty_survival);
    dva += (1.0 - assumptions.recovery_rate) * discount * profile.ene[index] *
           (own_survival - next_own_survival);
    fva += discount * assumptions.funding_spread * profile.epe[index] * step;
    counterparty_survival = next_counterparty_survival;
    own_survival = next_own_survival;
    previous = profile.times[index];
  }
  return {cva, dva, fva, std::move(profile)};
}

Money simplified_simm(double absolute_dv01, double absolute_vega, double rates_delta_weight,
                      double rates_vega_weight) {
  if (absolute_dv01 < 0.0 || absolute_vega < 0.0 || rates_delta_weight < 0.0 ||
      rates_vega_weight < 0.0) {
    throw ValidationError("SIMM inputs cannot be negative");
  }
  const double delta_margin = rates_delta_weight * absolute_dv01;
  const double vega_margin = rates_vega_weight * absolute_vega;
  return std::sqrt(delta_margin * delta_margin + vega_margin * vega_margin);
}

Money compute_mva(Money initial_margin, double funding_spread, Time horizon,
                  DiscountFactor average_discount) {
  if (initial_margin < 0.0 || funding_spread < 0.0 || horizon < 0.0 || average_discount <= 0.0) {
    throw ValidationError("Invalid MVA inputs");
  }
  return initial_margin * funding_spread * horizon * average_discount;
}

}  // namespace qf
