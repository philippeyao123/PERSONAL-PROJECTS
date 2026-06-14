"""Lexicon-based baselines: VADER and TextBlob.

These need no training and serve as a reference point the TF-IDF models should
comfortably beat on formal financial text.
"""
from __future__ import annotations

import logging

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .config import VADER_NEG_THRESHOLD, VADER_POS_THRESHOLD

log = logging.getLogger(__name__)


def _ensure_vader() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def classify_vader(compound: float) -> str:
    """Map a VADER compound score to a label."""
    if compound >= VADER_POS_THRESHOLD:
        return "positive"
    if compound <= VADER_NEG_THRESHOLD:
        return "negative"
    return "neutral"


def classify_textblob(polarity: float) -> str:
    """Map a TextBlob polarity score to a label."""
    if polarity > 0:
        return "positive"
    if polarity < 0:
        return "negative"
    return "neutral"


def run_vader(df: pd.DataFrame, text_col: str = "news") -> pd.DataFrame:
    """Add ``vader_compound`` and ``vader_sentiment`` columns."""
    log.info("Running VADER")
    _ensure_vader()
    sid = SentimentIntensityAnalyzer()
    out = df.copy()
    out["vader_compound"] = out[text_col].apply(lambda x: sid.polarity_scores(x)["compound"])
    out["vader_sentiment"] = out["vader_compound"].apply(classify_vader)
    return out


def run_textblob(df: pd.DataFrame, text_col: str = "news") -> pd.DataFrame:
    """Add ``textblob_polarity`` and ``textblob_sentiment`` columns."""
    from textblob import TextBlob

    log.info("Running TextBlob")
    out = df.copy()
    out["textblob_polarity"] = out[text_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    out["textblob_sentiment"] = out["textblob_polarity"].apply(classify_textblob)
    return out
