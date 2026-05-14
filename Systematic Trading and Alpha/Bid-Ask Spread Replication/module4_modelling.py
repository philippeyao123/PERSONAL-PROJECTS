"""
Module 4 — Spread Modelling
============================
Regresses reconstructed bid-ask spread on microstructure features.
Three estimators compared:
    1. OLS (baseline, interpretable)
    2. Ridge Regression (regularized, handles collinearity)
    3. XGBoost (nonlinear, captures regime interactions)

Walk-forward cross-validation to respect time-series structure (no leakage).

Author: Philippe-Emmanuel Yao | MSc Financial Mathematics, LSE
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass


# ============================================================================
# Linear Models (no sklearn dependency — pure numpy OLS/Ridge)
# ============================================================================

def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares: β = (X'X)^{-1} X'y"""
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        return np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression: β = (X'X + αI)^{-1} X'y"""
    n_feat = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(n_feat)
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


class LinearSpreadModel:
    """OLS or Ridge spread model with standardization."""

    def __init__(self, method: str = 'ols', alpha: float = 1.0):
        assert method in ('ols', 'ridge')
        self.method = method
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSpreadModel':
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-10
        Xs = (X - self.mu_) / self.sigma_
        Xs = np.c_[np.ones(len(Xs)), Xs]   # Add intercept column

        if self.method == 'ols':
            beta = ols_fit(Xs, y)
        else:
            beta = ridge_fit(Xs, y, self.alpha)

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu_) / self.sigma_
        return self.intercept_ + Xs @ self.coef_

    def coef_table(self, feature_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.coef_,
            'abs_coef': np.abs(self.coef_),
        }).sort_values('abs_coef', ascending=False)


# ============================================================================
# XGBoost Gradient Boosted Trees
# ============================================================================

class SimpleGBM:
    """
    Minimal gradient boosted regression trees (MSE loss).
    Pure numpy implementation — no external ML library required.

    At each iteration t:
        F_t(x) = F_{t-1}(x) + η * h_t(x)
    where h_t is a regression tree fit on residuals r_i = y_i - F_{t-1}(x_i)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 10,
        subsample: float = 0.8,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.seed = seed
        self.trees_: List = []
        self.F0_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleGBM':
        rng = np.random.default_rng(self.seed)
        self.F0_ = np.mean(y)
        F = np.full(len(y), self.F0_)
        self.trees_ = []

        for _ in range(self.n_estimators):
            residuals = y - F
            # Subsample rows
            n_sample = max(1, int(len(y) * self.subsample))
            idx = rng.choice(len(y), n_sample, replace=False)
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Impurity-based feature importance (sum of MSE reductions)."""
        importances = np.zeros(len(feature_names))
        for tree in self.trees_:
            importances += tree.feature_importances_
        importances /= importances.sum() + 1e-10
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False)


class RegressionTree:
    """Minimal CART regression tree."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.feature_importances_ = None
        self._n_features = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionTree':
        self._n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_features)
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n = len(y)
        leaf_val = np.mean(y)

        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return {'leaf': True, 'value': leaf_val}

        best_feat, best_thresh, best_gain = None, None, -np.inf
        parent_mse = np.var(y) * n

        for feat in range(X.shape[1]):
            thresholds = np.percentile(X[:, feat], np.linspace(10, 90, 10))
            for thresh in np.unique(thresholds):
                left  = y[X[:, feat] <= thresh]
                right = y[X[:, feat] >  thresh]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                gain = parent_mse - (np.var(left) * len(left) + np.var(right) * len(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        if best_feat is None:
            return {'leaf': True, 'value': leaf_val}

        self.feature_importances_[best_feat] += best_gain
        mask = X[:, best_feat] <= best_thresh
        return {
            'leaf': False,
            'feat': best_feat,
            'thresh': best_thresh,
            'left':  self._build(X[mask],  y[mask],  depth + 1),
            'right': self._build(X[~mask], y[~mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.tree_) for x in X])


# ============================================================================
# Walk-Forward Cross Validation
# ============================================================================

def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict,
    n_folds: int = 5,
    train_frac: float = 0.6,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation.

    Folds:
        [Train | Test] → [Train + Test_1 | Test_2] → ...

    Returns
    -------
    DataFrame of out-of-sample metrics per model per fold.
    """
    n = len(y)
    min_train = int(n * train_frac)
    fold_size = (n - min_train) // n_folds

    results = []
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end  = train_end + fold_size
        if test_end > n:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]

        for name, model_cls in models.items():
            model = model_cls()
            model.fit(X_tr, y_tr)
            y_hat = model.predict(X_te)
            y_hat = np.maximum(y_hat, 0)   # Spreads ≥ 0

            mae  = np.mean(np.abs(y_te - y_hat))
            rmse = np.sqrt(np.mean((y_te - y_hat) ** 2))
            ss_res = np.sum((y_te - y_hat) ** 2)
            ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)

            results.append({
                'fold': fold + 1,
                'model': name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'train_n': train_end,
                'test_n': fold_size,
            })

    return pd.DataFrame(results)


# ============================================================================
# Main Training Pipeline
# ============================================================================

@dataclass
class SpreadModelResult:
    models: Dict
    feature_names: List[str]
    cv_results: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: Dict[str, np.ndarray]


def train_spread_models(feats: pd.DataFrame, l2_spread: pd.Series) -> SpreadModelResult:
    """
    Full spread modelling pipeline.

    Features: lagged VPIN, OFI, Kyle λ, Realized Vol
    Target: L2 spread (reconstructed, not true — realistic inference setting)
    """
    feature_cols = ['vpin_lag1', 'ofi_lag1', 'kyle_lambda_lag1', 'realized_vol_lag1']
    available = [c for c in feature_cols if c in feats.columns]

    # Align features with L2 spread (may have different index)
    df_model = feats[['timestamp'] + available].copy()
    df_model['bucket'] = df_model['timestamp'].dt.floor('30s')
    df_model = df_model.groupby('bucket')[available].mean()

    l2 = l2_spread.copy()
    l2.index = pd.to_datetime(l2.index)

    # Inner join on bucket timestamps
    combined = df_model.join(l2.rename('target'), how='inner').dropna()

    X = combined[available].values
    y = combined['target'].values

    # Train/test split (80/20, no shuffle)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Model factory
    def make_ols():   return LinearSpreadModel('ols')
    def make_ridge(): return LinearSpreadModel('ridge', alpha=0.5)
    def make_gbm():   return SimpleGBM(n_estimators=80, max_depth=3, learning_rate=0.05)

    model_factories = {'OLS': make_ols, 'Ridge': make_ridge, 'GBM': make_gbm}

    # Walk-forward CV
    cv_results = walk_forward_cv(X, y, model_factories)

    # Final fit on full train set
    models = {}
    predictions = {}
    for name, factory in model_factories.items():
        m = factory()
        m.fit(X_tr, y_tr)
        models[name] = m
        pred = np.maximum(m.predict(X_te), 0)
        predictions[name] = pred

    return SpreadModelResult(
        models=models,
        feature_names=available,
        cv_results=cv_results,
        X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te,
        predictions=predictions,
    )


if __name__ == '__main__':
    from module1_data import simulate_tick_data, lee_ready_classify, MarketConfig
    from module2_reconstruction import level2_proxy
    from module3_features import build_feature_matrix

    cfg = MarketConfig()
    df = simulate_tick_data(cfg)
    df['lr_direction'] = lee_ready_classify(df)
    feats = build_feature_matrix(df)
    l2 = level2_proxy(df).set_index('bucket_time')['spread_proxy']

    res = train_spread_models(feats, l2)

    print("Walk-Forward CV Results (mean across folds):")
    print(res.cv_results.groupby('model')[['mae', 'rmse', 'r2']].mean().round(5))

    print("\nGBM Feature Importances:")
    print(res.models['GBM'].feature_importance(res.feature_names).to_string(index=False))
