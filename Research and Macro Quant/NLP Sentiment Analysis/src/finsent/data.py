"""Loading and cleaning of the Financial PhraseBank dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import LABEL_ORDER

log = logging.getLogger(__name__)


def load_data(file_path: str | Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load the Financial PhraseBank CSV.

    The original file has no header: column 0 is the label, column 1 the text.
    """
    file_path = Path(file_path)
    log.info("Loading data from: %s", file_path)
    df = pd.read_csv(file_path, encoding=encoding, header=None, names=["sentiment", "news"])
    log.info("Loaded %d rows", len(df))
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset.

    1. Drop exact duplicates and rows with missing values.
    2. Strip whitespace and lower-case the labels.
    3. Drop any row whose label is not in ``LABEL_ORDER``.
    """
    initial = len(df)
    df = df.drop_duplicates().dropna().copy()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["news"] = df["news"].str.strip()

    valid = set(LABEL_ORDER)
    bad = df[~df["sentiment"].isin(valid)]
    if len(bad):
        log.warning("Dropping %d rows with unrecognised labels", len(bad))
        df = df[df["sentiment"].isin(valid)]

    df = df[df["news"].str.len() > 0]
    log.info("After cleaning: %d rows (removed %d)", len(df), initial - len(df))
    return df.reset_index(drop=True)


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return label counts in canonical order."""
    return df["sentiment"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)


def load_and_clean(file_path: str | Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Convenience wrapper: load + preprocess in one call."""
    return preprocess_data(load_data(file_path, encoding))
