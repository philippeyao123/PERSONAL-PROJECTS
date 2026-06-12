"""
Machine Learning for Option Pricing — corrected end-to-end pipeline
====================================================================

Fixes vs. original project:
1. TARGET: bid-ask midpoint normalised by strike (C/K), not raw lastPrice.
   - Removes bid-ask noise in the label.
   - Exploits Black-Scholes homogeneity C(S,K) = K * c(S/K, T, sigma, r),
     so errors are comparable across cheap OTM and expensive ITM options.
2. IV CIRCULARITY: per-option implied vol is computed FROM the price, so
   feeding it back to predict the price is near-circular. We use the
   per-expiry ATM implied vol as the vol input instead: it is a smile-level
   anchor, and the model must learn the smile (skew/curvature) itself.
   A "circular" feature set with per-option IV is kept for comparison so the
   gap can be quantified and discussed honestly.
3. SPLIT: options from the same expiry share the same smile snapshot, so a
   random row split leaks. We split and cross-validate by expiry group
   (GroupKFold / GroupShuffleSplit) — the correct cross-sectional analogue
   of walk-forward.
4. SCALING: StandardScaler only inside a sklearn Pipeline (fit on train
   folds only) and only for scale-sensitive models (Linear, SVR, MLP).
   Tree models receive raw features. The target is never standardised:
   normalisation by K already puts it on a comparable scale.
5. BASELINE: plain Black-Scholes (not Garman-Kohlhagen, which is FX).
6. METRICS: RMSE / MAE / R^2 on C/K, RMSE back in price units, and relative
   error by moneyness bucket — all written to results CSV.
7. NO-ARBITRAGE: intrinsic/upper bounds + isotonic monotonicity in strike
   per expiry, applied as post-processing; calendar-spread violations are
   measured and reported.

Run:
    python ml_option_pricing.py --ticker AAPL          # live data (yfinance)
    python ml_option_pricing.py --synthetic            # offline smoke test
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
IMAGES_DIR = "images"
RESULTS_CSV = "results.csv"


# ----------------------------------------------------------------------------
# 1. Black-Scholes analytics
# ----------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    """European call under Black-Scholes (no dividends). Vectorised."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, sigma):
    T = np.maximum(np.asarray(T, float), 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, lo=1e-4, hi=5.0, tol=1e-8, max_iter=100):
    """Bisection IV (robust where Newton diverges for deep ITM/OTM)."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-12 or price >= S:
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = bs_call(S, K, T, r, mid) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# 2. Data: live (yfinance) or synthetic smile generator
# ----------------------------------------------------------------------------
def fetch_live_chain(ticker: str, r: float = 0.045, max_expiries: int = 16) -> pd.DataFrame:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    rows = []
    for expiry in tk.options[:max_expiries]:
        chain = tk.option_chain(expiry).calls
        T = (pd.Timestamp(expiry) - pd.Timestamp.now()).days / 365.0
        if T < 7 / 365:          # avoid expiry-day microstructure noise
            continue
        chain = chain.assign(expiry=expiry, T=T, S=spot, r=r)
        rows.append(chain)
    df = pd.concat(rows, ignore_index=True)

    # Quality filters: live two-sided quotes, sane spreads
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["ask"] - df["bid"]) / df["mid"] < 0.5]          # spread < 50%
    df["moneyness"] = df["S"] / df["strike"]
    df = df[df["moneyness"].between(0.7, 1.3)]                  # liquid wing cut
    return df[["expiry", "strike", "S", "T", "r", "mid", "bid", "ask"]]


def synthetic_chain(n_expiries: int = 10, n_strikes: int = 35,
                    spot: float = 100.0, r: float = 0.045) -> pd.DataFrame:
    """SVI-like smile + bid-ask noise: lets the pipeline run offline and
    gives a ground truth against which leakage/overfit is easy to see."""
    rows = []
    for i in range(n_expiries):
        T = (15 + 40 * i) / 365.0
        expiry = f"T{i:02d}"
        atm_vol = 0.22 + 0.04 * np.exp(-3 * T)                  # term structure
        for K in np.linspace(0.72 * spot, 1.28 * spot, n_strikes):
            k = np.log(K / spot)
            sigma = atm_vol + 0.35 * k**2 / np.sqrt(T) - 0.12 * k  # smile + skew
            sigma = float(np.clip(sigma, 0.05, 1.5))
            true_px = float(bs_call(spot, K, T, r, sigma))
            spread = max(0.02, 0.01 * true_px)
            noise = RNG.normal(0, spread / 4)
            mid = max(true_px + noise, 0.01)
            rows.append(dict(expiry=expiry, strike=K, S=spot, T=T, r=r,
                             mid=mid, bid=mid - spread / 2, ask=mid + spread / 2))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_moneyness"] = np.log(df["S"] / df["strike"])
    df["sqrt_T"] = np.sqrt(df["T"])

    # Per-option IV (circular feature — for comparison only)
    df["iv"] = [implied_vol(p, s, k, t, r) for p, s, k, t, r in
                zip(df["mid"], df["S"], df["strike"], df["T"], df["r"])]
    df = df.dropna(subset=["iv"])

    # Per-expiry ATM IV: smile-level anchor, far less circular.
    # The model must learn skew/curvature, not just invert BS.
    atm = (df.assign(d=lambda x: (x["strike"] - x["S"]).abs())
             .sort_values("d").groupby("expiry").first()["iv"]
             .rename("atm_iv"))
    df = df.merge(atm, on="expiry")

    # Targets and BS anchors — everything normalised by K (homogeneity)
    df["y"] = df["mid"] / df["strike"]
    df["bs_atm"] = bs_call(df["S"], df["strike"], df["T"], df["r"], df["atm_iv"])
    df["bs_atm_norm"] = df["bs_atm"] / df["strike"]
    df["resid"] = df["y"] - df["bs_atm_norm"]        # residual-mode target
    return df


CLEAN_FEATURES = ["log_moneyness", "sqrt_T", "atm_iv", "r", "bs_atm_norm"]
CIRCULAR_FEATURES = ["log_moneyness", "sqrt_T", "iv", "r"]


# ----------------------------------------------------------------------------
# 4. Models — scaling only where it belongs, fitted inside CV folds
# ----------------------------------------------------------------------------
def make_models(fast: bool) -> dict:
    scaled = lambda est: Pipeline([("scaler", StandardScaler()), ("model", est)])
    models = {
        "Linear": scaled(LinearRegression()),
        "SVR_RBF": scaled(SVR(C=10.0, gamma="scale", epsilon=1e-3)),
        "RandomForest": RandomForestRegressor(
            n_estimators=100 if fast else 400, max_depth=None,
            min_samples_leaf=3, n_jobs=-1, random_state=42),
        "GradBoost": HistGradientBoostingRegressor(
            max_iter=150 if fast else 600, learning_rate=0.06,
            max_depth=6, l2_regularization=1.0, random_state=42),
        "MLP": scaled(MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            alpha=1e-4, learning_rate_init=1e-3, batch_size=64,
            max_iter=80 if fast else 500, early_stopping=True,
            n_iter_no_change=10, random_state=42)),
    }
    return models


def tune_svr(X, y, groups, fast: bool):
    """Grid search with GROUP-aware CV — the scaler refits inside each fold,
    so no statistics leak across expiries."""
    if fast:
        return None
    pipe = Pipeline([("scaler", StandardScaler()), ("model", SVR(epsilon=1e-3))])
    grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.1, 1.0]}
    gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=4),
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


# ----------------------------------------------------------------------------
# 5. No-arbitrage post-processing
# ----------------------------------------------------------------------------
def enforce_no_arbitrage(df_test: pd.DataFrame, pred_norm: np.ndarray) -> np.ndarray:
    """Bounds + strike monotonicity (isotonic, per expiry).
    Calendar violations are measured in evaluate()."""
    px = pred_norm * df_test["strike"].values
    S, K, T, r = (df_test[c].values for c in ["S", "strike", "T", "r"])

    px = np.maximum(px, np.maximum(S - K * np.exp(-r * T), 0.0))   # lower bound
    px = np.minimum(px, S)                                          # upper bound

    out = px.copy()
    tmp = df_test.assign(px=px)
    for _, g in tmp.groupby("expiry"):
        if len(g) < 3:
            continue
        order = g["strike"].argsort().values
        idx = g.index[order]
        iso = IsotonicRegression(increasing=False)                  # call px decreasing in K
        out[df_test.index.get_indexer(idx)] = iso.fit_transform(
            g["strike"].values[order], g["px"].values[order])
    return out / K


def calendar_violation_rate(df_test: pd.DataFrame, pred_norm: np.ndarray) -> float:
    """Share of (K, T1<T2) pairs where C(T2) < C(T1) — should be ~0."""
    tmp = df_test.assign(px=pred_norm * df_test["strike"].values)
    tmp["kb"] = tmp["strike"].round(0)
    viol = tot = 0
    for _, g in tmp.groupby("kb"):
        g = g.sort_values("T")
        if len(g) > 1:
            d = np.diff(g["px"].values)
            viol += int((d < -1e-6).sum())
            tot += len(d)
    return viol / tot if tot else 0.0


# ----------------------------------------------------------------------------
# 6. Evaluation
# ----------------------------------------------------------------------------
@dataclass
class EvalRow:
    model: str
    mode: str
    rmse_norm: float
    mae_norm: float
    r2: float
    rmse_price: float
    rel_err_otm: float
    rel_err_atm: float
    rel_err_itm: float
    cal_viol: float


def bucket_rel_err(df_test, pred_px):
    rel = np.abs(pred_px - df_test["mid"].values) / df_test["mid"].values
    m = df_test["log_moneyness"].values
    return (float(np.median(rel[m < -0.05])) if (m < -0.05).any() else np.nan,
            float(np.median(rel[np.abs(m) <= 0.05])) if (np.abs(m) <= 0.05).any() else np.nan,
            float(np.median(rel[m > 0.05])) if (m > 0.05).any() else np.nan)


def evaluate(name, mode, df_test, pred_norm) -> EvalRow:
    pred_norm = enforce_no_arbitrage(df_test, pred_norm)
    y = df_test["y"].values
    pred_px = pred_norm * df_test["strike"].values
    otm, atm, itm = bucket_rel_err(df_test, pred_px)
    return EvalRow(name, mode,
                   float(np.sqrt(mean_squared_error(y, pred_norm))),
                   float(mean_absolute_error(y, pred_norm)),
                   float(r2_score(y, pred_norm)),
                   float(np.sqrt(mean_squared_error(df_test["mid"], pred_px))),
                   otm, atm, itm,
                   calendar_violation_rate(df_test, pred_norm))


# ----------------------------------------------------------------------------
# 7. Main experiment
# ----------------------------------------------------------------------------
def run(df: pd.DataFrame, fast: bool = False):
    df = build_features(df)
    groups = df["expiry"].values

    # Hold out entire expiries — the cross-sectional analogue of walk-forward
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(df, groups=groups))
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    print(f"Train: {len(train)} options / {train['expiry'].nunique()} expiries | "
          f"Test: {len(test)} options / {test['expiry'].nunique()} expiries")

    results: list[EvalRow] = []
    best = {"rmse": np.inf, "pred_norm": None, "name": None}

    # Baseline: BS priced with the ATM vol (no smile) — the bar to beat
    results.append(evaluate("BS_ATMvol", "baseline", test,
                            test["bs_atm_norm"].values))

    for feats, tag in [(CLEAN_FEATURES, "clean"), (CIRCULAR_FEATURES, "circular_IV")]:
        Xtr, Xte = train[feats].values, test[feats].values
        for name, model in make_models(fast).items():
            # Direct mode: predict C/K
            model.fit(Xtr, train["y"].values)
            results.append(evaluate(name, f"direct[{tag}]", test, model.predict(Xte)))

            # Residual mode: predict (C - BS_atm)/K, add anchor back
            model.fit(Xtr, train["resid"].values)
            pred = test["bs_atm_norm"].values + model.predict(Xte)
            row = evaluate(name, f"residual[{tag}]", test, pred)
            results.append(row)
            if tag == "clean" and row.rmse_price < best["rmse"]:
                best.update(rmse=row.rmse_price, name=name,
                            pred_norm=enforce_no_arbitrage(test, pred))

    # SHAP on the best tree model (clean features, residual mode)
    shap_vals = None
    try:
        import shap
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                            max_depth=6, random_state=42)
        gbm.fit(train[CLEAN_FEATURES], train["resid"])
        sv = shap.Explainer(gbm, train[CLEAN_FEATURES])(test[CLEAN_FEATURES])
        shap_vals = sv.values
        imp = pd.Series(np.abs(sv.values).mean(0), index=CLEAN_FEATURES)
        print("\nSHAP |mean| importance (residual mode, clean features):")
        print(imp.sort_values(ascending=False).to_string())
    except ImportError:
        print("\n[shap not installed — skipping interpretability step]")

    res = pd.DataFrame([vars(r) for r in results]).sort_values("rmse_price")
    res.to_csv(RESULTS_CSV, index=False)
    return res, df, train, test, best, shap_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fast", action="store_true", help="small models, smoke test")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    df = synthetic_chain() if args.synthetic else fetch_live_chain(args.ticker)
    print(f"Loaded {len(df)} call options")

    res, df_feat, train, test, best, shap_vals = run(df, fast=args.fast)
    pd.set_option("display.float_format", lambda v: f"{v:,.5f}")
    print("\n===== RESULTS (sorted by RMSE in price units) =====")
    print(res.to_string(index=False))
    print(f"\nSaved -> {RESULTS_CSV}")

    if not args.no_plots and best["pred_norm"] is not None:
        import plots
        best_px = best["pred_norm"] * test["strike"].values
        plots.fig_pipeline()
        plots.fig_model_comparison(res)
        plots.fig_surface_error(df_feat, test, best_px)
        plots.fig_residuals(test, best_px)
        if shap_vals is not None:
            plots.fig_shap(shap_vals, CLEAN_FEATURES)
        print(f"Figures -> {IMAGES_DIR}/  (best model: {best['name']}, "
              f"RMSE ${best['rmse']:.3f})")


if __name__ == "__main__":
    main()
