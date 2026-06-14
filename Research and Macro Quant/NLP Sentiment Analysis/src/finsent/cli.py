"""Command-line interface for the financial-sentiment pipeline.

Examples
--------
    python -m finsent.cli train --data-path data/all-data.csv
    python -m finsent.cli train --imbalance-strategy smote --no-lexicon
    python -m finsent.cli predict --model artifacts/best_sentiment_model.pkl \\
        --text "Shares plunged after the profit warning."
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .inference import predict_sentiment
from .pipeline import run_pipeline


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="finsent", description="Financial news sentiment pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- train -------------------------------------------------------------
    t = sub.add_parser("train", help="Train and evaluate the full pipeline")
    t.add_argument("--data-path", default="data/all-data.csv")
    t.add_argument("--encoding", default="ISO-8859-1")
    t.add_argument("--artifacts-dir", default="artifacts")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--cv-folds", type=int, default=5)
    t.add_argument(
        "--imbalance-strategy",
        choices=["class_weight", "smote", "none"],
        default="class_weight",
    )
    t.add_argument("--refit-metric", default="f1_macro",
                   choices=["f1_macro", "f1_weighted", "accuracy"])
    t.add_argument("--models", nargs="+", default=["logreg", "linear_svm", "random_forest"])
    t.add_argument("--random-state", type=int, default=42)
    t.add_argument("--no-lexicon", action="store_true", help="Skip VADER/TextBlob baselines")
    t.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    t.add_argument("--log-level", default="INFO")

    # ---- predict -----------------------------------------------------------
    pr = sub.add_parser("predict", help="Predict sentiment for one or more strings")
    pr.add_argument("--model", default="artifacts/best_sentiment_model.pkl")
    pr.add_argument("--text", action="append", required=True, help="Headline (repeatable)")
    pr.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.log_level)

    if args.command == "train":
        cfg = Config(
            data_path=args.data_path,
            encoding=args.encoding,
            artifacts_dir=args.artifacts_dir,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            imbalance_strategy=args.imbalance_strategy,
            refit_metric=args.refit_metric,
            models=tuple(args.models),
            random_state=args.random_state,
        )
        run_pipeline(cfg, run_lexicon=not args.no_lexicon, make_plots=not args.no_plots)

    elif args.command == "predict":
        out = predict_sentiment(args.text, args.model)
        print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
