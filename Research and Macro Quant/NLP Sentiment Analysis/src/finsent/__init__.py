"""finsent — modular three-class sentiment analysis for financial news.

Public API:
    >>> from finsent import Config, run_pipeline, predict_sentiment
"""
from .config import Config, LABEL_ORDER
from .pipeline import run_pipeline
from .inference import predict_sentiment, load_model, save_model

__version__ = "1.0.0"
__all__ = [
    "Config",
    "LABEL_ORDER",
    "run_pipeline",
    "predict_sentiment",
    "load_model",
    "save_model",
]
