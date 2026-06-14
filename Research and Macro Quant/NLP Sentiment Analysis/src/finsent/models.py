"""TF-IDF + classifier pipelines, grid search, and training orchestration.

Each model is a scikit-learn ``Pipeline`` so the vectoriser is fit only on the
training fold inside cross-validation — no leakage. Class imbalance is handled
according to ``Config.imbalance_strategy``.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import LinearSVC

from .config import Config

log = logging.getLogger(__name__)

# Map our config refit_metric to a sklearn scorer string.
_SCORERS = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
}


def _make_pipeline(estimator, use_smote: bool):
    """Wrap a TF-IDF + estimator pipeline, inserting SMOTE if requested.

    Uses imbalanced-learn's pipeline when SMOTE is active so resampling happens
    correctly inside each CV fold.
    """
    steps = [("tfidf", TfidfVectorizer())]
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)
    steps.append(("clf", estimator))
    return SkPipeline(steps)


def build_pipelines(cfg: Config) -> dict[str, tuple[Any, dict]]:
    """Return ``{name: (pipeline, param_grid)}`` for the models in ``cfg``."""
    use_smote = cfg.imbalance_strategy == "smote"
    # class_weight only when we are NOT resampling, otherwise we double-correct
    cw = "balanced" if cfg.imbalance_strategy == "class_weight" else None

    tfidf_grid = {
        "tfidf__max_features": list(cfg.tfidf_max_features),
        "tfidf__ngram_range": list(cfg.tfidf_ngram_ranges),
        "tfidf__sublinear_tf": [True],
    }

    registry: dict[str, tuple[Any, dict]] = {
        "logreg": (
            _make_pipeline(
                LogisticRegression(class_weight=cw, max_iter=1000, random_state=cfg.random_state),
                use_smote,
            ),
            {**tfidf_grid, "clf__C": list(cfg.logreg_C)},
        ),
        "linear_svm": (
            _make_pipeline(
                LinearSVC(class_weight=cw, max_iter=5000, random_state=cfg.random_state),
                use_smote,
            ),
            {**tfidf_grid, "clf__C": list(cfg.svm_C)},
        ),
        "random_forest": (
            _make_pipeline(
                RandomForestClassifier(
                    class_weight=cw, n_jobs=-1, random_state=cfg.random_state
                ),
                use_smote,
            ),
            {**tfidf_grid, "clf__n_estimators": list(cfg.rf_n_estimators)},
        ),
    }
    return {name: registry[name] for name in cfg.models if name in registry}


def split_data(df: pd.DataFrame, cfg: Config):
    """Stratified train/test split on the text and label columns."""
    return train_test_split(
        df["news"],
        df["sentiment"],
        test_size=cfg.test_size,
        stratify=df["sentiment"],
        random_state=cfg.random_state,
    )


def train_models(df: pd.DataFrame, cfg: Config):
    """Grid-search every configured model.

    Returns
    -------
    best_models : dict[str, Pipeline]      fitted best estimator per model
    fitted_search : dict[str, GridSearchCV] the search objects (for inspection)
    X_test, y_test                          held-out split for evaluation
    """
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scorer = _SCORERS.get(cfg.refit_metric, "f1_macro")

    best_models: dict[str, Any] = {}
    searches: dict[str, GridSearchCV] = {}

    for name, (pipe, grid) in build_pipelines(cfg).items():
        log.info("Grid-searching %s", name)
        gs = GridSearchCV(pipe, grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        log.info("%s best CV %s = %.4f", name, scorer, gs.best_score_)
        best_models[name] = gs.best_estimator_
        searches[name] = gs

    return best_models, searches, X_test, y_test
