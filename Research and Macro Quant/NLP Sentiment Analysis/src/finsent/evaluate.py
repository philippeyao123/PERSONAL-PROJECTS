"""Evaluation metrics and plots, consistent across lexicon and ML models."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; no display needed for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import LABEL_ORDER

log = logging.getLogger(__name__)


def evaluate(y_true, y_pred, model_name: str, verbose: bool = True) -> dict:
    """Accuracy + macro/weighted/per-class F1 as a flat metrics dict."""
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)
    f1_cls = f1_score(y_true, y_pred, average=None, labels=LABEL_ORDER, zero_division=0)

    if verbose:
        print(f"\n{'=' * 55}\n  {model_name}\n{'=' * 55}")
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  F1 (macro):    {f1_mac:.4f}")
        print(f"  F1 (weighted): {f1_wt:.4f}\n")
        print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "f1_negative": f1_cls[0],
        "f1_neutral": f1_cls[1],
        "f1_positive": f1_cls[2],
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: Path | None = None) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=120)
        log.info("Saved %s", path)
    plt.close(fig)


def plot_model_comparison(results: list[dict], out_dir: Path | None = None) -> None:
    df = pd.DataFrame(results).set_index("model")
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    fig, ax = plt.subplots(figsize=(9, 5))
    df[metrics].plot.bar(ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "model_comparison.png"
        fig.savefig(path, dpi=120)
        log.info("Saved %s", path)
    plt.close(fig)


def plot_class_distribution(dist: pd.Series, out_dir: Path | None = None) -> None:
    """Bar chart of label counts (illustrates the neutral-heavy imbalance)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"negative": "#d1495b", "neutral": "#9aa0a6", "positive": "#2e8b57"}
    ax.bar(dist.index, dist.values, color=[colors.get(k, "#4c72b0") for k in dist.index])
    for i, v in enumerate(dist.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution — Financial PhraseBank")
    ax.set_ylim(0, dist.values.max() * 1.12)
    fig.tight_layout()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "class_distribution.png"
        fig.savefig(path, dpi=120)
        log.info("Saved %s", path)
    plt.close(fig)


def results_table(results: list[dict]) -> pd.DataFrame:
    cols = ["accuracy", "f1_macro", "f1_weighted", "f1_negative", "f1_neutral", "f1_positive"]
    return pd.DataFrame(results).set_index("model")[cols].round(4)
