"""Test suite. Focus on the properties that matter most for credibility:
no look-ahead, correct dollar-neutrality, and sane diagnostic behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factory.backtest.costs import CostParams, TransactionCostModel
from alpha_factory.data.loader import PITDataLoader, make_synthetic_panel
from alpha_factory.diagnostics.metrics import (
    deflated_sharpe_ratio,
    performance_stats,
    probabilistic_sharpe_ratio,
)
from alpha_factory.factors.combiner import neutralize, winsorize, zscore
from alpha_factory.factors.library import Momentum
from alpha_factory.portfolio.construction import (
    apply_position_limits,
    quantile_long_short,
)


# ----------------------------- data / PIT -----------------------------
def test_pit_lag_prevents_lookahead():
    """A fundamental reported at date D must not be visible before D+lag."""
    dates = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.DataFrame(100.0, index=dates, columns=["X"])
    fund = pd.DataFrame({"X": np.arange(200.0)}, index=dates)
    loader = PITDataLoader(fundamental_lag_days=90)
    panel = loader.from_frames(prices, {"f": fund})
    # On any date, the visible fundamental value's original timestamp must be
    # at least 90 days earlier.
    as_of = dates[150]
    visible = panel.fundamentals["f"].loc[as_of, "X"]
    # value at as_of should correspond to a row whose original date <= as_of-90d
    original_date_idx = int(visible)
    assert dates[original_date_idx] <= as_of - pd.Timedelta(days=90)


def test_synthetic_panel_keeps_delisted():
    panel = make_synthetic_panel(n_assets=100, n_days=600, delisting_rate=0.1)
    # Some columns must contain NaNs (delisted) but remain in the universe.
    assert len(panel.universe) == 100
    assert panel.prices.isna().any().any()


# ----------------------------- factors -----------------------------
def test_momentum_sign():
    panel = make_synthetic_panel(n_assets=50, n_days=400, seed=1)
    mom = Momentum(lookback=252, skip=21)
    s = mom.compute(panel, panel.dates[-1])
    assert not s.empty
    assert s.index.isin(panel.universe).all()


def test_factor_only_uses_past_data():
    """Computing at date t must be invariant to data after t."""
    panel = make_synthetic_panel(n_assets=50, n_days=500, seed=2)
    t = panel.dates[300]
    mom = Momentum(lookback=126, skip=21)
    s_full = mom.compute(panel, t)
    truncated = panel.slice(panel.dates[0], t)
    s_trunc = mom.compute(truncated, t)
    pd.testing.assert_series_equal(s_full.sort_index(), s_trunc.sort_index())


# ----------------------------- transforms -----------------------------
def test_zscore_properties():
    s = pd.Series([1.0, 2, 3, 4, 5])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_winsorize_clips_tails():
    s = pd.Series(list(range(100)) + [10_000.0])
    w = winsorize(s, 0.05)
    assert w.max() < 10_000


def test_neutralize_zeroes_group_means():
    s = pd.Series([1.0, 3, 10, 12], index=["a", "b", "c", "d"])
    g = pd.Series(["X", "X", "Y", "Y"], index=["a", "b", "c", "d"])
    n = neutralize(s, g)
    assert abs(n[["a", "b"]].mean()) < 1e-9
    assert abs(n[["c", "d"]].mean()) < 1e-9


# ----------------------------- portfolio -----------------------------
def test_quantile_book_is_dollar_neutral():
    signal = pd.Series(np.random.default_rng(0).normal(size=200))
    w = quantile_long_short(signal, quantile=0.1)
    assert abs(w.sum()) < 1e-9           # dollar neutral
    assert abs(w.abs().sum() - 1.0) < 1e-9  # unit gross


def test_position_limits_respected():
    signal = pd.Series(np.random.default_rng(0).normal(size=50))
    w = quantile_long_short(signal, quantile=0.1)
    capped = apply_position_limits(w, max_weight=0.05)
    assert capped.abs().max() <= 0.05 + 1e-9


# ----------------------------- costs -----------------------------
def test_turnover_zero_when_unchanged():
    w = pd.Series({"a": 0.5, "b": -0.5})
    cm = TransactionCostModel(CostParams())
    assert cm.turnover(w, w) == pytest.approx(0.0)


def test_cost_increases_with_turnover():
    cm = TransactionCostModel(CostParams())
    w0 = pd.Series({"a": 0.5, "b": -0.5})
    w1 = pd.Series({"a": -0.5, "b": 0.5})  # full flip
    assert cm.cost(w0, w1) > cm.cost(w0, w0)


# ----------------------------- diagnostics -----------------------------
def test_deflated_sharpe_decreases_with_trials():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.04, 120))
    d_few = deflated_sharpe_ratio(r, n_trials=2)["dsr"]
    d_many = deflated_sharpe_ratio(r, n_trials=500)["dsr"]
    assert d_many <= d_few


def test_psr_in_unit_interval():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.01, 0.04, 120))
    p = probabilistic_sharpe_ratio(r)
    assert 0.0 <= p <= 1.0


def test_performance_stats_basic():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.005, 0.02, 240))
    stats = performance_stats(r, periods_per_year=12)
    assert stats.ann_vol > 0
    assert -1.0 <= stats.max_drawdown <= 0.0


# ----------------------------- integration -----------------------------
def test_full_pipeline_runs_and_finds_planted_signal():
    """Smoke test: pipeline runs end-to-end and value factors lead on IC."""
    from alpha_factory.data.loader import make_synthetic_panel
    from alpha_factory.pipeline import run_pipeline

    panel = make_synthetic_panel(n_assets=200, n_days=1200, seed=7)
    out = run_pipeline(panel=panel, n_trials=20, combine_method="ic")

    assert out["net_stats"].ann_vol > 0
    assert not out["ic_summary"].empty
    # The planted signal lives in the value fundamentals -> they should have
    # the strongest mean IC of the factor set.
    ic = out["ic_summary"]["ic_mean"].sort_values(ascending=False)
    assert ic.index[0].startswith("value_")


def test_combiner_methods_all_produce_signal():
    from alpha_factory.factors.combiner import FactorCombiner

    rng = np.random.default_rng(11)
    fs = {
        "f1": pd.Series(rng.normal(size=100), index=[f"A{i}" for i in range(100)]),
        "f2": pd.Series(rng.normal(size=100), index=[f"A{i}" for i in range(100)]),
    }
    for method in ("equal", "ic", "ridge"):
        c = FactorCombiner(method=method)
        sig = c.combine(fs, ic_weights={"f1": 0.5, "f2": 0.5})
        assert not sig.empty


# ----------------------------- equity loader -----------------------------
def test_equity_loader_uses_cache(tmp_path):
    """If a cache exists, the loader must read it without network access."""
    import pandas as pd

    from alpha_factory.data.equity import EquityDataLoader

    # Fabricate a tiny cache.
    root = tmp_path / "eq"
    root.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(
        100.0, index=dates, columns=["AAA", "BBB", "CCC"]
    ).cumsum()
    adv = pd.DataFrame(1e8, index=dates, columns=["AAA", "BBB", "CCC"])
    meta = pd.DataFrame({"sector": ["TECH", "FIN", "ENER"]},
                        index=["AAA", "BBB", "CCC"])
    prices.to_parquet(root / "prices.parquet")
    adv.to_parquet(root / "dollar_adv.parquet")
    meta.to_parquet(root / "metadata.parquet")

    loader = EquityDataLoader(cache_root=root)
    panel, dollar_adv = loader.load(use_cache=True)
    assert len(panel.universe) == 3
    assert dollar_adv.shape[1] == 3
    assert "sector" in panel.metadata


# ----------------------------- tsmom replication -----------------------------
def test_tsmom_runs_on_synthetic_proxies():
    """TSMOM study runs end-to-end and returns a period breakdown."""
    import numpy as np
    import pandas as pd

    from alpha_factory.diagnostics.tsmom_replication import TimeSeriesMomentum

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2006-01-01", periods=3000)
    cols = ["SPY", "IEF", "GLD", "DBC"]
    # Random-walk-ish prices with mild trend persistence.
    rets = rng.normal(0.0003, 0.01, (3000, len(cols)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                          index=dates, columns=cols)
    res = TimeSeriesMomentum().run(prices)
    assert np.isfinite(res.gross_sharpe)
    assert np.isfinite(res.net_sharpe)
    assert res.net_sharpe <= res.gross_sharpe + 1e-9  # costs never help
    assert "period" in res.by_period.columns
    assert (res.by_period["period"] == "in-sample (<=2012)").any()


def test_tsmom_costs_reduce_sharpe():
    import numpy as np
    import pandas as pd

    from alpha_factory.diagnostics.tsmom_replication import TimeSeriesMomentum

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2008-01-01", periods=2500)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (2500, 3)), axis=0)),
        index=dates, columns=["A", "B", "C"],
    )
    free = TimeSeriesMomentum(cost_bps=0.0).run(prices).net_sharpe
    costly = TimeSeriesMomentum(cost_bps=50.0).run(prices).net_sharpe
    assert costly <= free + 1e-9


# ----------------------------- plots -----------------------------
def test_plots_generate_files(tmp_path):
    """Figure functions write valid PNGs without error."""
    import numpy as np
    import pandas as pd

    from alpha_factory.diagnostics.plots import (
        plot_capacity,
        plot_equity_curve,
        plot_factor_ic,
        plot_tsmom_decay,
    )

    dates = pd.bdate_range("2020-01-01", periods=50)
    g = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 50), index=dates)
    n = g - 0.0002
    p1 = plot_equity_curve(g, n, tmp_path / "eq.png", "test")
    assert p1.exists() and p1.stat().st_size > 0

    ic = pd.DataFrame({
        "ic_mean": [0.04, -0.01, 0.02],
        "ic_std": [0.1, 0.1, 0.1],
        "ic_ir": [0.4, -0.1, 0.2],
        "t_stat": [3.0, -0.5, 1.0],
        "n": [80, 80, 80],
    }, index=["value", "lowvol", "mom"])
    p2 = plot_factor_ic(ic, tmp_path / "ic.png")
    assert p2.exists()

    cap = pd.DataFrame({
        "aum": np.logspace(7, 11, 10),
        "net_return": np.linspace(0.06, -0.04, 10),
    })
    p3 = plot_capacity(cap, tmp_path / "cap.png")
    assert p3.exists()

    bp = pd.DataFrame({
        "period": ["in-sample (<=2012)", "2000s", "2010s", "2020s"],
        "sharpe": [0.4, 1.07, 0.17, 0.40],
        "n_months": [70, 34, 120, 60],
    })
    p4 = plot_tsmom_decay(bp, tmp_path / "decay.png")
    assert p4.exists()
