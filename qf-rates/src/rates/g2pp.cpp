#include "qf/rates/g2pp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "qf/core/error.hpp"
#include "qf/core/numerics.hpp"

namespace qf {
namespace {

constexpr std::array<double, 8> kHermiteNodes{
    -2.930637420257244, -1.981656756695843, -1.157193712446780, -0.381186990207322,
    0.381186990207322,  1.157193712446780,  1.981656756695843,  2.930637420257244};
constexpr std::array<double, 8> kHermiteWeights{
    0.000199604072211, 0.017077983007413, 0.207802325814892, 0.661147012558241,
    0.661147012558241, 0.207802325814892, 0.017077983007413, 0.000199604072211};

double ou_covariance(double speed_left, double speed_right, double volatility_left,
                     double volatility_right, double correlation, Time left, Time right) {
  const Time minimum = std::min(left, right);
  return correlation * volatility_left * volatility_right *
         std::exp(-speed_left * left - speed_right * right) *
         (std::exp((speed_left + speed_right) * minimum) - 1.0) / (speed_left + speed_right);
}

std::array<std::array<double, 3>, 3> cholesky3(
    const std::array<std::array<double, 3>, 3>& covariance) {
  std::array<std::array<double, 3>, 3> result{};
  for (std::size_t row = 0; row < 3U; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double value = covariance[row][column];
      for (std::size_t inner = 0; inner < column; ++inner) {
        value -= result[row][inner] * result[column][inner];
      }
      if (row == column) {
        result[row][column] = std::sqrt(std::max(value, 0.0));
      } else if (result[column][column] > 1.0e-16) {
        result[row][column] = value / result[column][column];
      }
    }
  }
  return result;
}

double swap_value_at_expiry(const G2ppModel& model, const EuropeanSwaption& swaption,
                            FactorState state) {
  const auto& periods = swaption.underlying_schedule.periods();
  const double terminal_bond = model.discount_bond(swaption.expiry, periods.back().end, state);
  double fixed_leg = 0.0;
  for (const auto& period : periods) {
    fixed_leg += period.accrual * model.discount_bond(swaption.expiry, period.payment, state);
  }
  const double payer_value = 1.0 - terminal_bond - swaption.strike * fixed_leg;
  return swaption.type == OptionType::Call ? payer_value : -payer_value;
}

}  // namespace

G2ppModel::G2ppModel(YieldCurvePtr curve, G2ppParameters parameters)
    : curve_(std::move(curve)), parameters_(parameters) {
  if (!curve_) {
    throw ValidationError("G2++ requires a yield curve");
  }
  if (!(parameters_.a > 0.0 && parameters_.b > 0.0 && parameters_.sigma >= 0.0 &&
        parameters_.eta >= 0.0 && parameters_.rho > -1.0 && parameters_.rho < 1.0)) {
    throw ValidationError("Invalid G2++ parameters");
  }
}

double G2ppModel::factor_loading(double mean_reversion, Time horizon) const {
  if (horizon < 0.0 || mean_reversion <= 0.0) {
    throw ValidationError("Factor loading requires positive speed and non-negative horizon");
  }
  return -std::expm1(-mean_reversion * horizon) / mean_reversion;
}

double G2ppModel::integrated_variance(Time horizon) const {
  if (horizon < 0.0) {
    throw ValidationError("Variance horizon cannot be negative");
  }
  const auto one_factor = [horizon](double speed, double volatility) {
    return volatility * volatility / (speed * speed) *
           (horizon + 2.0 * std::expm1(-speed * horizon) / speed -
            std::expm1(-2.0 * speed * horizon) / (2.0 * speed));
  };
  const auto& p = parameters_;
  const double cross =
      2.0 * p.rho * p.sigma * p.eta / (p.a * p.b) *
      (horizon + std::expm1(-p.a * horizon) / p.a + std::expm1(-p.b * horizon) / p.b -
       std::expm1(-(p.a + p.b) * horizon) / (p.a + p.b));
  return std::max(0.0, one_factor(p.a, p.sigma) + one_factor(p.b, p.eta) + cross);
}

std::array<std::array<double, 2>, 2> G2ppModel::factor_covariance(Time horizon) const {
  if (horizon < 0.0) {
    throw ValidationError("Covariance horizon cannot be negative");
  }
  const auto& p = parameters_;
  const double variance_x = p.sigma * p.sigma * (-std::expm1(-2.0 * p.a * horizon)) / (2.0 * p.a);
  const double variance_y = p.eta * p.eta * (-std::expm1(-2.0 * p.b * horizon)) / (2.0 * p.b);
  const double covariance =
      p.rho * p.sigma * p.eta * (-std::expm1(-(p.a + p.b) * horizon)) / (p.a + p.b);
  return {{{variance_x, covariance}, {covariance, variance_y}}};
}

DiscountFactor G2ppModel::discount_bond(Time observation, Time maturity, FactorState state) const {
  if (!(observation >= 0.0 && maturity >= observation)) {
    throw ValidationError("Bond times require 0 <= observation <= maturity");
  }
  if (maturity == observation) {
    return 1.0;
  }
  const auto& p = parameters_;
  const double horizon = maturity - observation;
  const double adjustment = 0.5 * (integrated_variance(horizon) - integrated_variance(maturity) +
                                   integrated_variance(observation));
  const double a_value =
      curve_->discount(maturity) / curve_->discount(observation) * std::exp(adjustment);
  return a_value *
         std::exp(-factor_loading(p.a, horizon) * state.x - factor_loading(p.b, horizon) * state.y);
}

Rate G2ppModel::short_rate(Time observation, FactorState state) const {
  constexpr double epsilon = 1.0e-5;
  const double deterministic =
      -std::log(curve_->discount(observation + epsilon) / curve_->discount(observation)) / epsilon +
      0.5 * (integrated_variance(observation + epsilon) - integrated_variance(observation)) /
          epsilon;
  return deterministic + state.x + state.y;
}

FactorState G2ppModel::evolve(FactorState state, Time step, double normal_x,
                              double normal_independent) const {
  if (step < 0.0) {
    throw ValidationError("G2++ evolution step cannot be negative");
  }
  const auto& p = parameters_;
  const double decay_x = std::exp(-p.a * step);
  const double decay_y = std::exp(-p.b * step);
  const double standard_x = p.sigma * std::sqrt((-std::expm1(-2.0 * p.a * step)) / (2.0 * p.a));
  const double standard_y = p.eta * std::sqrt((-std::expm1(-2.0 * p.b * step)) / (2.0 * p.b));
  const double increment_covariance =
      p.rho * p.sigma * p.eta * (-std::expm1(-(p.a + p.b) * step)) / (p.a + p.b);
  const double increment_correlation =
      standard_x > 0.0 && standard_y > 0.0
          ? std::clamp(increment_covariance / (standard_x * standard_y), -1.0, 1.0)
          : 0.0;
  const double correlated_y =
      increment_correlation * normal_x +
      std::sqrt(std::max(0.0, 1.0 - increment_correlation * increment_correlation)) *
          normal_independent;
  return {state.x * decay_x + standard_x * normal_x, state.y * decay_y + standard_y * correlated_y};
}

Money g2pp_european_swaption(const G2ppModel& model, const EuropeanSwaption& swaption,
                             std::size_t quadrature_order) {
  if (!(swaption.expiry > 0.0 && swaption.notional > 0.0) ||
      swaption.underlying_schedule.start() < swaption.expiry - 1.0e-12) {
    throw ValidationError("Invalid European swaption definition");
  }
  if (quadrature_order != 8U) {
    throw ValidationError("This release provides the validated order-8 Gauss-Hermite rule");
  }
  const auto factor_covariance = model.factor_covariance(swaption.expiry);
  const auto& p = model.parameters();
  const auto covariance_with_integral = [&](bool with_x) {
    const auto covariance_at_time = [&](double time) {
      if (with_x) {
        return ou_covariance(p.a, p.a, p.sigma, p.sigma, 1.0, time, swaption.expiry) +
               ou_covariance(p.b, p.a, p.eta, p.sigma, p.rho, time, swaption.expiry);
      }
      return ou_covariance(p.a, p.b, p.sigma, p.eta, p.rho, time, swaption.expiry) +
             ou_covariance(p.b, p.b, p.eta, p.eta, 1.0, time, swaption.expiry);
    };
    return adaptive_simpson(covariance_at_time, 0.0, swaption.expiry, 1.0e-11, 16);
  };
  const std::array<std::array<double, 3>, 3> covariance{{
      {model.integrated_variance(swaption.expiry), covariance_with_integral(true),
       covariance_with_integral(false)},
      {covariance_with_integral(true), factor_covariance[0][0], factor_covariance[0][1]},
      {covariance_with_integral(false), factor_covariance[1][0], factor_covariance[1][1]},
  }};
  const auto lower = cholesky3(covariance);
  const double deterministic_discount = model.curve().discount(swaption.expiry) *
                                        std::exp(-0.5 * model.integrated_variance(swaption.expiry));
  double expectation = 0.0;
  constexpr double standard_normal_scale = 1.4142135623730950488;
  for (std::size_t first = 0; first < kHermiteNodes.size(); ++first) {
    for (std::size_t second = 0; second < kHermiteNodes.size(); ++second) {
      for (std::size_t third = 0; third < kHermiteNodes.size(); ++third) {
        const std::array<double, 3> independent{standard_normal_scale * kHermiteNodes[first],
                                                standard_normal_scale * kHermiteNodes[second],
                                                standard_normal_scale * kHermiteNodes[third]};
        std::array<double, 3> joint{};
        for (std::size_t row = 0; row < 3U; ++row) {
          for (std::size_t column = 0; column <= row; ++column) {
            joint[row] += lower[row][column] * independent[column];
          }
        }
        const double payoff =
            std::max(swap_value_at_expiry(model, swaption, {joint[1], joint[2]}), 0.0);
        expectation += kHermiteWeights[first] * kHermiteWeights[second] * kHermiteWeights[third] *
                       std::exp(-joint[0]) * payoff;
      }
    }
  }
  const double normalization = std::pow(std::numbers::pi, -1.5);
  return swaption.notional * deterministic_discount * normalization * expectation;
}

}  // namespace qf
