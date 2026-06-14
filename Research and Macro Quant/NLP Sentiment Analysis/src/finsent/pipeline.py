"""End-to-end orchestration tying every stage together."""
from __future__ import annotations

import json
import logging

from . import data as data_mod
from . import lexicon, models
from .config import Config
from .evaluate import (
    evaluate,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    results_table,
)
from .inference import save_model

log = logging.getLogger(__name__)


def run_pipeline(cfg: Config, run_lexicon: bool = True, make_plots: bool = True):
    """Run the full pipeline and return a results dict.

    Stages: load/clean -> (optional) lexicon baselines -> ML grid search ->
    evaluation on held-out test set -> persist best model + metrics.
    """
    df = data_mod.load_and_clean(cfg.data_path, cfg.encoding)
    dist = data_mod.class_distribution(df)
    log.info("Class distribution:\n%s", dist.to_string())

    all_results: list[dict] = []
    plot_dir = cfg.artifacts_dir if make_plots else None
    if make_plots:
        plot_class_distribution(dist, plot_dir)

    # --- Lexicon baselines (evaluated on the full set, they are untrained) ---
    if run_lexicon:
        dfx = lexicon.run_textblob(lexicon.run_vader(df))
        all_results.append(evaluate(dfx["sentiment"], dfx["vader_sentiment"], "VADER"))
        all_results.append(evaluate(dfx["sentiment"], dfx["textblob_sentiment"], "TextBlob"))
        if make_plots:
            plot_confusion_matrix(dfx["sentiment"], dfx["vader_sentiment"], "VADER", plot_dir)
            plot_confusion_matrix(dfx["sentiment"], dfx["textblob_sentiment"], "TextBlob", plot_dir)

    # --- ML models ----------------------------------------------------------
    best_models, searches, X_test, y_test = models.train_models(df, cfg)
    ml_results = []
    for name, model in best_models.items():
        res = evaluate(y_test, model.predict(X_test), name)
        ml_results.append(res)
        all_results.append(res)
        if make_plots:
            plot_confusion_matrix(y_test, model.predict(X_test), name, plot_dir)

    if make_plots:
        plot_model_comparison(all_results, plot_dir)

    # --- Select & persist the best model by configured metric ---------------
    best_name = max(ml_results, key=lambda r: r[cfg.refit_metric])["model"]
    save_model(best_models[best_name], cfg.model_path)

    summary = results_table(all_results)
    print("\nFull results summary:\n", summary.to_string())
    print(f"\nBest model: {best_name} -> {cfg.model_path}")

    # Persist metrics + config for reproducibility
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "metrics.json").write_text(
        json.dumps({"results": all_results, "best_model": best_name, "config": cfg.to_dict()}, indent=2)
    )

    return {
        "summary": summary,
        "best_name": best_name,
        "best_models": best_models,
        "searches": searches,
        "results": all_results,
    }
