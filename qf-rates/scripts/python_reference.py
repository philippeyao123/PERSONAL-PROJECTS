#!/usr/bin/env python3
"""Independent standard-library reference values; QuantLib is optional."""

from math import erf, exp, log, pi, sqrt


def cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black76(forward: float, strike: float, vol: float, expiry: float, discount: float) -> float:
    std = vol * sqrt(expiry)
    d1 = (log(forward / strike) + 0.5 * std * std) / std
    return discount * (forward * cdf(d1) - strike * cdf(d1 - std))


def bachelier(forward: float, strike: float, vol: float, expiry: float, discount: float) -> float:
    std = vol * sqrt(expiry)
    d = (forward - strike) / std
    pdf = exp(-0.5 * d * d) / sqrt(2.0 * pi)
    return discount * ((forward - strike) * cdf(d) + std * pdf)


def main() -> None:
    values = {
        "flat_curve_discount_5y": exp(-0.03 * 5.0),
        "black76_atm": black76(100.0, 100.0, 0.20, 1.0, 0.95),
        "bachelier_atm": bachelier(0.03, 0.03, 0.01, 1.0, 0.97),
    }
    for name, value in values.items():
        print(f"{name}={value:.12f}")
    try:
        import QuantLib as ql  # type: ignore

        black = ql.blackFormula(ql.Option.Call, 100.0, 100.0, 0.20, 0.95)
        normal = ql.bachelierBlackFormula(ql.Option.Call, 0.03, 0.03, 0.01, 0.97)
        print(f"quantlib_black76_atm={black:.12f}")
        print(f"quantlib_bachelier_atm={normal:.12f}")
    except ImportError:
        print("QuantLib not installed: standard-library references completed")


if __name__ == "__main__":
    main()

