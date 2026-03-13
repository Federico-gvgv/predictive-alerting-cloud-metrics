"""Evaluation entry-point.

Usage::

    python -m src.eval --config configs/default.yaml

Loads the dataset, builds windows, loads the trained model from the
outputs directory, computes metrics, and saves a JSON evaluation report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import load_dataset
from src.data.splits import time_split
from src.data.windowing import create_windows
from src.utils.config import load_config, pretty_print_config
from src.utils.logging import get_logger, set_seed


def _load_model(run_dir: Path, cfg: dict):
    """Load a model from *run_dir* by reading its ``model_meta.json``."""
    meta_path = run_dir / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No model_meta.json found in {run_dir}. Run training first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_type = meta.get("model_type", "")

    if model_type == "heuristic":
        from src.models.heuristic import HeuristicModel
        return HeuristicModel.load(run_dir, cfg)

    if model_type == "sklearn":
        from src.models.logreg_baseline import SklearnBaseline
        return SklearnBaseline.load(run_dir, cfg)

    raise ValueError(f"Unknown model_type '{model_type}' in {meta_path}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a predictive-alerting model on cloud metrics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation pipeline."""
    args = parse_args(argv)
    cfg = load_config(args.config)

    # Reproducibility
    seed: int = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    logger = get_logger(__name__)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Resolved config:\n%s", pretty_print_config(cfg))

    # ── 1. Load dataset & windows ────────────────────────────────────
    df = load_dataset(cfg)
    logger.info("Raw dataset: %d rows.", len(df))

    W = cfg["windowing"]["W"]
    H = cfg["windowing"]["H"]
    stride = cfg["windowing"].get("stride", 1)

    X, y, timestamps = create_windows(df, W=W, H=H, stride=stride)

    split_cfg = cfg.get("split", {})
    splits = time_split(
        X,
        y,
        timestamps,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    test_X, test_y, test_ts = splits["test"]
    logger.info("Test split: %d windows, incident rate %.2f%%.",
                len(test_y), test_y.mean() * 100)

    # ── 2. Load trained model ────────────────────────────────────────
    model_choice = cfg["model"]["model_choice"]
    output_dir = Path(cfg.get("training", {}).get("output_dir", "outputs"))
    run_dir = output_dir / model_choice

    logger.info("Loading model from %s …", run_dir)
    model = _load_model(run_dir, cfg)

    # ── 3. Predict ───────────────────────────────────────────────────
    scores = model.predict_proba(test_X)

    # ── 4. Compute metrics ───────────────────────────────────────────
    eval_cfg = cfg.get("evaluation", {})
    threshold = eval_cfg.get("alert_threshold", 0.5)

    preds = (scores >= threshold).astype(int)

    results: dict[str, float] = {}

    # ROC-AUC & PR-AUC (require both classes present)
    if len(np.unique(test_y)) > 1:
        results["roc_auc"] = float(roc_auc_score(test_y, scores))
        results["pr_auc"] = float(average_precision_score(test_y, scores))
    else:
        logger.warning("Only one class in test set – AUC metrics undefined.")
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")

    results["precision"] = float(precision_score(test_y, preds, zero_division=0))
    results["recall"] = float(recall_score(test_y, preds, zero_division=0))
    results["f1"] = float(f1_score(test_y, preds, zero_division=0))
    results["threshold"] = threshold
    results["n_test"] = int(len(test_y))
    results["n_positive"] = int(test_y.sum())
    results["n_predicted_positive"] = int(preds.sum())

    logger.info("── Evaluation Results ──")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info("  %-24s %.4f", k, v)
        else:
            logger.info("  %-24s %s", k, v)

    # ── 5. Save report ───────────────────────────────────────────────
    report_path = run_dir / "eval_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved to %s.", report_path)

    # ------------------------------------------------------------------
    # TODO – event-style metrics:
    #   - Apply cooldown logic to predicted alerts
    #   - Match predicted alert events to ground-truth incident intervals
    #   - Compute event-level precision / recall / F1
    # ------------------------------------------------------------------
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
