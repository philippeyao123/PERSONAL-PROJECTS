#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "qf/rates/swap.hpp"

using Catch::Approx;

TEST_CASE("Vanilla swap par rate produces zero NPV") {
  const qf::FlatYieldCurve curve(0.025);
  const auto seed_swap = qf::make_vanilla_swap(0.0, 10.0, 0.0, 1'000'000.0);
  const double par = seed_swap.par_rate(curve, curve);
  const auto par_swap = qf::make_vanilla_swap(0.0, 10.0, par, 1'000'000.0);
  REQUIRE(par_swap.npv(curve, curve) == Approx(0.0).margin(1.0e-8));
  REQUIRE(par > 0.025);
}

TEST_CASE("Payer swap loses value when its fixed coupon rises") {
  const qf::FlatYieldCurve curve(0.03);
  const auto low_coupon = qf::make_vanilla_swap(0.0, 5.0, 0.02);
  const auto high_coupon = qf::make_vanilla_swap(0.0, 5.0, 0.04);
  REQUIRE(low_coupon.npv(curve, curve) > high_coupon.npv(curve, curve));
}

TEST_CASE("Single curve floating leg telescopes") {
  const qf::FlatYieldCurve curve(0.02);
  const auto swap = qf::make_vanilla_swap(0.0, 4.0, 0.0, 1.0);
  REQUIRE(swap.floating_leg_npv(curve, curve) == Approx(1.0 - curve.discount(4.0)).margin(1.0e-12));
}
