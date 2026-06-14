"""Test suite for the finsent package (pytest-compatible)."""
from __future__ import annotations

import pandas as pd
import pytest

from finsent import Config
from finsent.config import LABEL_ORDER
from finsent.data import preprocess_data, class_distribution
from finsent.lexicon import classify_vader, classify_textblob, run_vader, run_textblob
from finsent.models import build_pipelines, split_data
from finsent.evaluate import evaluate, results_table


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "sentiment": ["positive", "negative", "neutral", "positive"],
            "news": [
                "The company posted strong earnings, beating all forecasts.",
                "Shares collapsed after the profit warning shocked markets.",
                "The board held its annual general meeting.",
                "Revenue grew sharply on robust demand and margin expansion.",
            ],
        }
    )


# ---- lexicon mapping -------------------------------------------------------
@pytest.mark.parametrize("score,label", [(0.5, "positive"), (-0.5, "negative"), (0.0, "neutral")])
def test_classify_vader(score, label):
    assert classify_vader(score) == label


@pytest.mark.parametrize("pol,label", [(0.3, "positive"), (-0.3, "negative"), (0.0, "neutral")])
def test_classify_textblob(pol, label):
    assert classify_textblob(pol) == label


def test_run_vader_columns(sample_df):
    out = run_vader(sample_df)
    assert {"vader_compound", "vader_sentiment"}.issubset(out.columns)
    assert out["vader_sentiment"].isin(LABEL_ORDER).all()


def test_run_textblob_columns(sample_df):
    out = run_textblob(sample_df)
    assert {"textblob_polarity", "textblob_sentiment"}.issubset(out.columns)


# ---- data cleaning ---------------------------------------------------------
def test_preprocess_dedup_and_validate():
    df = pd.DataFrame(
        {"sentiment": ["POSITIVE ", "weird", "neutral"], "news": ["a", "b", "c"]}
    )
    out = preprocess_data(df)
    assert "weird" not in out["sentiment"].values
    assert "positive" in out["sentiment"].values  # stripped + lowercased


def test_class_distribution_order(sample_df):
    dist = class_distribution(sample_df)
    assert list(dist.index) == LABEL_ORDER


# ---- models ----------------------------------------------------------------
def test_build_pipelines_respects_config():
    cfg = Config(models=("logreg",))
    pipes = build_pipelines(cfg)
    assert set(pipes) == {"logreg"}
    pipe, grid = pipes["logreg"]
    assert "clf__C" in grid


def test_smote_strategy_inserts_step():
    pytest.importorskip("imblearn")
    cfg = Config(models=("logreg",), imbalance_strategy="smote")
    pipe, _ = build_pipelines(cfg)["logreg"]
    assert "smote" in dict(pipe.steps)


def test_split_is_stratified():
    # need >= 2 members per class for a stratified split
    df = pd.DataFrame(
        {
            "sentiment": ["positive", "negative", "neutral"] * 4,
            "news": [f"headline number {i}" for i in range(12)],
        }
    )
    cfg = Config(test_size=0.5, random_state=0)
    Xtr, Xte, ytr, yte = split_data(df, cfg)
    assert len(Xtr) + len(Xte) == len(df)
    # each class present in both splits
    assert set(ytr) == set(yte) == set(LABEL_ORDER)


# ---- evaluation ------------------------------------------------------------
def test_evaluate_perfect():
    y = ["positive", "negative", "neutral"]
    m = evaluate(y, y, "perfect", verbose=False)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_results_table_columns():
    res = [evaluate(["positive"], ["positive"], "m", verbose=False)]
    tbl = results_table(res)
    assert "f1_macro" in tbl.columns


def test_config_rejects_bad_strategy():
    with pytest.raises(ValueError):
        Config(imbalance_strategy="banana")
