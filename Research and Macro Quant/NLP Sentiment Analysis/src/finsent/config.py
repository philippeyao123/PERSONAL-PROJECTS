"""Central configuration for the financial-sentiment pipeline.

Every tunable lives here so experiments are reproducible and the rest of the
package stays free of magic numbers. Override any field at runtime by passing a
``Config`` instance, or from the CLI via ``--key value`` flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Canonical label order used everywhere (metrics, confusion matrices, plots).
# Keep it fixed so per-class columns always line up.
LABEL_ORDER: list[str] = ["negative", "neutral", "positive"]

# VADER compound-score cut-offs (the standard thresholds from the paper).
VADER_POS_THRESHOLD: float = 0.05
VADER_NEG_THRESHOLD: float = -0.05


@dataclass
class Config:
    """All pipeline settings in one immutable-ish container."""

    # ---- I/O -----------------------------------------------------------------
    data_path: Path = Path("data/all-data.csv")
    encoding: str = "ISO-8859-1"
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "best_sentiment_model.pkl"

    # ---- Reproducibility -----------------------------------------------------
    random_state: int = 42

    # ---- Train / test split --------------------------------------------------
    test_size: float = 0.2
    cv_folds: int = 5

    # ---- Class-imbalance handling -------------------------------------------
    # "class_weight"  -> balanced class weights inside each estimator
    # "smote"         -> SMOTE oversampling of the minority class in the pipeline
    # "none"          -> no special handling
    imbalance_strategy: str = "class_weight"

    # ---- TF-IDF / classifier grid (kept small enough to run on a laptop) -----
    tfidf_max_features: tuple[int, ...] = (5000, 10000)
    tfidf_ngram_ranges: tuple[tuple[int, int], ...] = ((1, 1), (1, 2))
    logreg_C: tuple[float, ...] = (0.1, 1.0, 10.0)
    svm_C: tuple[float, ...] = (0.1, 1.0, 5.0)
    rf_n_estimators: tuple[int, ...] = (200, 400)

    # ---- Which models to fit -------------------------------------------------
    models: tuple[str, ...] = ("logreg", "linear_svm", "random_forest")

    # ---- Scoring metric used to select the best CV model ---------------------
    refit_metric: str = "f1_macro"

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.imbalance_strategy not in {"class_weight", "smote", "none"}:
            raise ValueError(f"Unknown imbalance_strategy: {self.imbalance_strategy}")

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_path"] = str(self.data_path)
        d["artifacts_dir"] = str(self.artifacts_dir)
        return d
