"""Model persistence and inference helpers."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

log = logging.getLogger(__name__)


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model -> %s", path)


def load_model(path: str | Path):
    return joblib.load(Path(path))


def predict_sentiment(texts: list[str], model_path: str | Path) -> pd.DataFrame:
    """Predict sentiment for a list of strings.

    Adds a ``confidence`` column (max class probability) when the estimator
    exposes ``predict_proba`` — LinearSVC does not, so it is omitted there.
    """
    model = load_model(model_path)
    preds = model.predict(texts)
    out = pd.DataFrame({"news": texts, "sentiment": preds})

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        out["confidence"] = proba.max(axis=1)
    return out
