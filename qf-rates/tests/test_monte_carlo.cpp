#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "qf/rates/monte_carlo.hpp"
#include "qf/rates/options.hpp"
#include "qf/rates/swap.hpp"

using Catch::Approx;

TEST_CASE("Generic Monte-Carlo confidence interval contains the analytic value") {
  constexpr double forward = 100.0;
  constexpr double strike = 100.0;
  constexpr double volatility = 0.20;
  const auto payoff = [](double normal) {
    const double terminal =
        forward * std::exp(-0.5 * volatility * volatility + volatility * normal);
    return std::max(terminal - strike, 0.0);
  };
  const auto result = qf::monte_carlo_normal(payoff, {40000, 1, 77, true});
  const double analytic = qf::black76(qf::OptionType::Call, forward, strike, volatility, 1.0).price;
  REQUIRE(analytic >= result.confidence_low - 0.05);
  REQUIRE(analytic <= result.confidence_high + 0.05);
}

TEST_CASE("Antithetic and control variates measurably reduce sampling error") {
  constexpr double forward = 100.0;
  constexpr double strike = 100.0;
  constexpr double volatility = 0.20;
  const auto terminal = [](double normal) {
    return forward * std::exp(-0.5 * volatility * volatility + volatility * normal);
  };
  const auto payoff = [&](double normal) { return std::max(terminal(normal) - strike, 0.0); };

  const auto plain = qf::monte_carlo_normal(payoff, {40000, 1, 17, false});
  const auto antithetic = qf::monte_carlo_normal(payoff, {40000, 1, 17, true});
  const auto controlled = qf::monte_carlo_normal(payoff, {40000, 1, 17, false}, terminal, forward);

  CAPTURE(plain.standard_error, antithetic.standard_error, controlled.standard_error);
  REQUIRE(antithetic.standard_error < plain.standard_error);
  REQUIRE(controlled.standard_error < 0.75 * plain.standard_error);
}

TEST_CASE("G2++ quadrature and Monte-Carlo agree within simulation noise") {
  auto curve = std::make_shared<qf::FlatYieldCurve>(0.025);
  const qf::G2ppModel model(curve, {});
  const auto swap = qf::make_vanilla_swap(1.0, 5.0, 0.025);
  const qf::EuropeanSwaption option{1.0, qf::Schedule(1.0, 5.0, 1.0), swap.par_rate(*curve, *curve),
                                    1.0, qf::OptionType::Call};
  const double deterministic = qf::g2pp_european_swaption(model, option);
  const auto simulated = qf::g2pp_european_swaption_mc(model, option, {30000, 96, 99, true});
  REQUIRE(std::abs(simulated.price - deterministic) < 5.0 * simulated.standard_error + 2.0e-4);
}
